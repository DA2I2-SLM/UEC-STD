import torch
import torch.nn as nn
from torch import optim
import joblib
import os
import time
import warnings
import numpy as np
import math
import random
import copy

from tqdm import tqdm
import wandb

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visualm
from utils.metrics import metric
from utils.dtw_metric import accelerated_dtw
from torch.utils.data import TensorDataset, DataLoader, random_split
from .utils_ECM import create_error_corrector, moving_avg


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.ori_pred_len = self.args.pred_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, criterion_name="MSE"):
        if criterion_name == "MAE":
            criterion = nn.L1Loss()
        elif criterion_name == "MSE":
            criterion = nn.MSELoss()
        elif criterion_name == "SmoothL1":
            criterion = nn.SmoothL1Loss()   
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
        return criterion

    def _resolve_backbone_setting(self, setting):
        """Return (backbone_setting, model_pred_len) with pred_len set to seq_len."""
        model_pred_len = self.args.seq_len
        parts = setting.split("_")
        parts[5] = str(model_pred_len)
        parts[11] = "pl" + str(model_pred_len)
        return "_".join(parts), model_pred_len

    def _backbone_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """Single forward pass through the backbone, handling AMP and attention output."""
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if getattr(self.args, 'output_attention', False):
            out = out[0]
        return out

    def _make_dec_inp(self, batch_y):
        """Build the zero-padded decoder input tensor."""
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        return torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

    def _inverse_transform(self, data_set, arr):
        """Inverse-scale arr if the dataset uses scaling."""
        if data_set.scale and self.args.inverse:
            shape = arr.shape
            arr = data_set.inverse_transform(arr.reshape(shape[0] * shape[1], -1)).reshape(shape)
        return arr

    def _candidate_error_coeffs(self, fallback):
        """Return error-coefficient candidates from CLI args or a fallback list."""
        error_flags = getattr(self.args, "error_flags", None)
        if isinstance(error_flags, str) and error_flags.strip():
            return [float(x) for x in error_flags.split(",") if x.strip()]
        if error_flags:
            return [float(x) for x in error_flags]
        return list(fallback)

    def _compute_dtw(self, preds, trues):
        """Compute mean DTW distance; returns 'Not calculated' if disabled."""
        if not self.args.use_dtw:
            return 'Not calculated'
        manhattan = lambda x, y: np.abs(x - y)
        dtw_list = []
        for i in range(preds.shape[0]):
            if i % 100 == 0:
                print(f"calculating dtw iter: {i}")
            d, _, _, _ = accelerated_dtw(preds[i].reshape(-1, 1), trues[i].reshape(-1, 1), dist=manhattan)
            dtw_list.append(d)
        return np.array(dtw_list).mean()

    def _save_results(self, setting, folder_path, preds, trues):
        """Compute metrics, append to summary txt, and save npy arrays."""
        dtw = self._compute_dtw(preds, trues)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, dtw:{dtw}')
        os.makedirs(folder_path, exist_ok=True)
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(f"{setting}  \n")
            f.write(f'mse:{mse}, mae:{mae}, dtw:{dtw}\n\n')
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        return mae, mse, rmse, mape, mspe, dtw



    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(criterion_name="MSE")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._make_dec_inp(batch_y)

                # encoder - decoder
                outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion, error_coeff=0, is_torch_model=False, error_model=None):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._make_dec_inp(batch_y)
                # encoder - decoder
                outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if error_coeff == 0:
                    loss = criterion(pred, true)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                    meinput = torch.cat([batch_x, outputs], dim=-1)

                    if is_torch_model:
                        perr = error_model(meinput)

                    else:
                        perr = error_model.predict(meinput.cpu().numpy())
                        perr = torch.tensor(perr).to(self.device)

                    outputs_pred = outputs + perr*error_coeff

                    loss = criterion(outputs_pred.to("cpu"), true)

                total_loss.append(loss.item())
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._make_dec_inp(batch_y)
                # encoder - decoder
                outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'

        self._save_results(setting, folder_path, preds, trues)

        return
    
    def test_infer(self, setting, test=0, ecm="linear"):
        setting, model_pred_len = self._resolve_backbone_setting(setting)
        print(setting)

        test_data, test_loader = self._get_data(flag='test')
        data_pred_len = self.args.pred_len
        num_ar = math.ceil(data_pred_len / model_pred_len)

        self.args.pred_len = model_pred_len

        if test:
            print('loading model')
            self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './scratch/infer_results/' + setting + "-" + ecm + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                obatch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                obatch_y_mark = batch_y_mark.float().to(self.device)

                oinput = batch_x[:, :, :]
                opreds = []
                otrues = []

                for j in range(num_ar):
                    batch_y = obatch_y[:, self.args.label_len + model_pred_len * j:self.args.label_len + model_pred_len * (j + 1), :]
                    batch_y_mark = obatch_y_mark[:, self.args.label_len + model_pred_len * j:self.args.label_len + model_pred_len * (j + 1), :]

                    if j == 0:
                        enc_inp = batch_x
                    else:
                        enc_inp = pred_y

                    if self.args.use_ar:
                        batch_x = enc_inp

                    dec_inp = self._make_dec_inp(batch_y)
                    outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    pred_y = outputs

                    batch_x_mark = batch_y_mark
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    opreds.append(outputs)
                    otrues.append(batch_y)

                opreds = np.concatenate(opreds, axis=1)
                otrues = np.concatenate(otrues, axis=1)
                opreds = opreds[:, :data_pred_len, :]
                otrues = otrues[:, :data_pred_len, :]

                pred = opreds
                true = otrues

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_ = oinput.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_.shape
                        input_ = test_data.inverse_transform(input_.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gts = []
                    pds = []
                    for ii in range(min(7, true.shape[-1])):
                        gt = np.concatenate((input_[0, :, ii], true[0, :, ii]), axis=0)
                        gts.append(gt)
                        pd = np.concatenate((input_[0, :, ii], pred[0, :, ii]), axis=0)
                        pds.append(pd)
                    visualm(gts, pds, os.path.join(folder_path, str(i) + '.pdf'))

                if i == 200:
                    break

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './scratch/infer_results/' + setting + '-' + ecm + '/'
        self._save_results(setting, folder_path, preds, trues)

        return
    def train_infer_batch(self, setting, ecm="linear", error_flags=None, loss_ecm="SmoothL1"):
        """
        Run inference using the backbone model, collect predictions and true values,
        train an error corrector model using prediction residuals, and search for the best error correction coefficient.

        Parameters:
        -----------
        setting : str
            The experiment configuration string used to identify checkpoints and results.
        ecm : str
            The error corrector model type. Supported types include 'linear', 'CNN', 'RNN', etc.
        error_flags : list[float]
            A list of error correction coefficients to search through.

        Steps:
        ------
        1. Load model and inference on validation data in an autoregressive fashion.
        2. Collect prediction residuals and inputs for error correction model.
        3. Train the error correction model on residuals.
        4. Evaluate and select best error coefficient on training data.
        5. Save the best-performing error corrector model.
        """
        # --- Setup and Load Backbone Model ---
        setting, model_pred_len = self._resolve_backbone_setting(setting)
        print(setting)

        # Load test data
        test_data, test_loader = self._get_data(flag='val')
        train_data, _ = self._get_data(flag='train')
        total_len = len(test_data)
        train_len = int(0.7 * total_len)
        val_len = total_len - train_len

        # Split
        _, ecm_val_data = torch.utils.data.random_split(test_data, [train_len, val_len])
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=True)
        ecm_val_loader = torch.utils.data.DataLoader(ecm_val_data, batch_size=self.args.batch_size, shuffle=False)
        
        more_data_len = train_len

        if len(train_data) >= more_data_len:
            generator = torch.Generator().manual_seed(42)
            more_data, _ = torch.utils.data.random_split(train_data, [more_data_len, len(train_data) - more_data_len], generator=generator)
        else:
            raise ValueError(f"train_data too small ({len(train_data)}) to extract {val_len} samples.")
        
        ecm_val_data = torch.utils.data.ConcatDataset([ecm_val_data, more_data])

        ecm_val_loader = torch.utils.data.DataLoader(ecm_val_data, batch_size=self.args.batch_size, shuffle=False)


        data_pred_len = self.args.pred_len
        num_ar = math.ceil(data_pred_len/model_pred_len)
        print("Num_ar", num_ar)

        # Override pred_len for backbone model prediction (typically to 96)
        self.args.pred_len = model_pred_len

        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        print(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'))

        # Load backbone model
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        # -------------------------------------------------------------------------------------- Inference Phase with Backbone Model --------------------------------------------------------------------------------------
        preds = []
        trues = []
        folder_path = './scratch/infer_results/' + setting + f'-' + ecm +  '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        correction_inputs = []
        correction_targets = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                obatch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                obatch_y_mark = batch_y_mark.float().to(self.device)

                for j in range(num_ar):
                    # Extract correct prediction window for autoregressive step
                    batch_y = obatch_y[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    batch_y_mark = obatch_y_mark[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    if j==0:
                        enc_inp = batch_x
                        oinput = batch_x
                    else:
                        enc_inp =  pred_y

                    batch_x = enc_inp #Autoregression
                    
                    # Decoder input
                    dec_inp = self._make_dec_inp(batch_y)

                    # Model prediction
                    outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs = outputs[:,:,:]
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]

                    pred_y = outputs

                    # Optional teacher forcing
                    if self.args.use_ar==0:
                        pred_y = obatch_y[:, self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]

                    batch_x_mark = batch_y_mark
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    vbatch_y = batch_y[:, :, f_dim:]
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
            
            
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs[:batch_y.shape[0], :batch_y.shape[1], :]
                    true = batch_y

                    if true.shape[1] < model_pred_len:
                        continue

                    # Store correction data (input = [input, pred], target = error)
                    correction_inputs.append(torch.cat([batch_x, pred_y[:, -self.args.pred_len:, f_dim:].to(self.device)], dim=-1))
                    correction_targets.append(torch.tensor(true-pred).to(self.device))

                    preds.append(pred)
                    trues.append(true)

                    # Visual inspection
                    if i % 10 == 0:
                        input =  batch_x[:,:,:].detach().cpu().numpy()
                        oinput = oinput.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                            oinput = test_data.inverse_transform(oinput.reshape(shape[0] * shape[1], -1)).reshape(shape)

                        gts = []
                        pds = []

                        for ii in range(min(7, true.shape[-1])):
                            gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                            gts.append(gt)
                            pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                            pds.append(pd)
                        visualm(gts, pds, os.path.join(folder_path, f"ar{self.args.use_ar}-{i}-{j}.pdf"))
                    oinput = vbatch_y
                    if i == 20000:
                        break
        # -------------------------------------------------------------------------------------- Evaluation & Save Predictions --------------------------------------------------------------------------------------
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './scratch/infer_results/' + setting + f'-' + ecm +  '/'

        self._save_results(setting, folder_path, preds, trues)
        # -------------------------------------------------------------------------------------- Prepare Correction Dataset --------------------------------------------------------------------------------------
        X = torch.cat(correction_inputs, dim=0)  # shape: (N, T, D)
        Y = torch.cat(correction_targets, dim=0)
        print(X.shape)
        print(Y.shape)
        # Normalize inputs (pred) and targets (error)
        Y_mean = Y.mean(dim=(0, 1), keepdim=True)
        Y_std = Y.std(dim=(0, 1), keepdim=True) + 1e-6
        Y = (Y - Y_mean) / Y_std

        dataset = TensorDataset(X, Y)
        train_size = int(0.7 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size   # 20% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # -------------------------------------------------------------------------------------- Train Error Corrector Model --------------------------------------------------------------------------------------
        input_dim=X.shape[-1]
        # Create the error corrector model
        modelerr, optimizer, loss_fn, scheduler, is_torch_model = create_error_corrector(ecm=ecm,
                                                                                        input_dim=input_dim,
                                                                                        T=X.shape[1],
                                                                                        output_dim=Y.shape[-1],
                                                                                        hidden_dim=self.args.err_h,
                                                                                        device=self.device,
                                                                                        loss_type=loss_ecm
                                                                                        )

        if is_torch_model:
            best_model_err = None
            best_val_loss = float('inf')  # Initialize best validation loss as infinity

            for epoch in range(100):
                modelerr.train()
                total_loss = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    pred = modelerr(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                
                avg_loss = total_loss / len(train_loader.dataset)
                scheduler.step(avg_loss)
                print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")
                
                # Validation phase
                modelerr.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = modelerr(xb)
                        loss = loss_fn(pred, yb)

                        val_loss += loss.item() * xb.size(0)
                
                avg_val_loss = val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

                # Check if the current validation loss is the best we've seen so far
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # print(f"New best validation loss: {best_val_loss:.4f}.")
                    best_model_err = copy.deepcopy(modelerr)
        else:
            # Collect all training data
            train_X, train_Y = [], []
            for xb, yb in train_loader:
                train_X.append(xb.cpu().numpy())
                train_Y.append(yb.cpu().numpy())
            train_X = np.concatenate(train_X, axis=0)
            train_Y = np.concatenate(train_Y, axis=0)

            # Fit the random forest model
            modelerr.fit(train_X, train_Y)
            best_model_err = copy.deepcopy(modelerr)

        for criterion_name in ["MAE", "MSE"]:
            print("Searching for best error coefficient using criterion:", criterion_name)
            criterion = self._select_criterion(criterion_name=criterion_name)
            best_i = 0
            best_val_loss = float('inf')  # Initialize best validation loss as infinity
            print("Begin searching for best error coefficient")
            for i in error_flags:
                print(i)
                avg_val_loss = self.vali(ecm_val_data, ecm_val_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model=best_model_err)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_i = i
                    print(f"New best validation loss: {best_val_loss:.4f} with coef {i}.")

            if is_torch_model:
                torch.save(best_model_err.state_dict(), os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-{criterion_name}.pth'))
            else:
                save_path = os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-{criterion_name}.pkl')
                joblib.dump(best_model_err, save_path)

    def test_infer_batch(self, setting, ecm="linear"):
        """
        Perform inference on test data using a backbone model with optional error correction.

        Parameters:
        -----------
        setting : str
            Experiment identifier used to load models and save results.
        ecm : str
            Type of error corrector model used (e.g., 'linear', 'mlp', 'rf').
        """
        # -------------------------------------------------------------------------------------- Load backbone model configuration ----------------------------------------------------------------------------------------
        setting, model_pred_len = self._resolve_backbone_setting(setting)
        print(setting)

        # Load test data
        test_data, test_loader = self._get_data(flag='test')
        data_pred_len =self.args.pred_len
        num_ar = math.ceil(data_pred_len/model_pred_len)

        # Update model prediction length to match training time
        self.args.pred_len = model_pred_len

        # Load backbone model
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        # -------------------------------------------------------------------------------------- Load trained error corrector model --------------------------------------------------------------------------------------

        user_errcor_coef = self.args.errcor_coef

        input_dim=self.args.enc_in*2
        print('loading modelerr')
        modelerr, _, _, _, is_torch_model = create_error_corrector(ecm=ecm,
                                                                    input_dim=input_dim,
                                                                    T=model_pred_len,
                                                                    output_dim=self.args.enc_in,
                                                                    hidden_dim=self.args.err_h,
                                                                    device=self.device,
                                                                    )
        
        # Try loading error corrector model using various saved coefficients
        possible_coeffs = self._candidate_error_coeffs([0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0])
        model_loaded = False
        loaded_any_criterion = False
        for criterion_name in ["MAE", "MSE"]:
            model_loaded = False
            loaded_coeff = None
            for coeff in possible_coeffs:
                try:
                    if is_torch_model:
                        print(os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pth'))
                        modelerr.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pth')))
                        modelerr.eval()
                        model_loaded = True
                        loaded_coeff = coeff
                        print("Modelerr loaded successfully.")
                        break
                    else:
                        print(os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pth'))
                        load_path = os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pkl')
                        modelerr = joblib.load(load_path)
                        model_loaded = True
                        loaded_coeff = coeff
                        print("Modelerr loaded successfully.")
                        break
                except FileNotFoundError:
                    continue
            if not model_loaded:
                print(f"No checkpoint found for criterion {criterion_name}; trying the next criterion.")
                continue

            if user_errcor_coef == -1:
                self.args.errcor_coef = loaded_coeff
                print(f"Auto mode: using coefficient {loaded_coeff} from training checkpoint.")
            else:
                self.args.errcor_coef = user_errcor_coef
                print(f"Manual mode: using coefficient {user_errcor_coef} (checkpoint was trained with {loaded_coeff}).")

            loaded_any_criterion = True

            # -------------------------------------------------------------------------------------- Begin inference --------------------------------------------------------------------------------------
            preds = []
            trues = []
            folder_path = './scratch/infer_results/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            self.model.eval()

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                    batch_x = batch_x.float().to(self.device)
                    obatch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    obatch_y_mark = batch_y_mark.float().to(self.device)
                    opreds = []
                    otrues = []
                    for j in range(num_ar):
                        batch_y = obatch_y[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                        batch_y_mark = obatch_y_mark[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]

                        # Set encoder input
                        if j==0:
                            enc_inp = batch_x
                            oinput = batch_x
                        else:
                            enc_inp =  pred_y
                        batch_x = enc_inp #Autoregression
                        
                        # Decoder input
                        dec_inp = self._make_dec_inp(batch_y)

                        # Run prediction
                        outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0

                        
                        # ----------------------------------
                        # Apply error correction if enabled
                        # ----------------------------------
                        if self.args.errcor_coef>0:
                            meinput = torch.cat([batch_x, outputs[:, -self.args.pred_len:, f_dim:]], dim=-1)
                            if is_torch_model:
                                perr = modelerr(meinput)
                            else:
                                perr = modelerr.predict(meinput.cpu().numpy())
                                perr = torch.tensor(perr).to(self.device)

                            outputs_pred = outputs + perr*self.args.errcor_coef


                        outputs = outputs[:, -self.args.pred_len:, :]

                        pred_y = outputs

                        # Optional teacher forcing
                        if self.args.use_ar==0:
                            pred_y = obatch_y[:, self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]

                        batch_x_mark = batch_y_mark
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        vbatch_y = batch_y[:, :, f_dim:]
                        outputs = outputs.detach().cpu().numpy()
                        batch_y = batch_y.detach().cpu().numpy()

                        # Inverse scale if necessary
                        if test_data.scale and self.args.inverse:
                            shape = outputs.shape
                            outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                            batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                
                        # Apply error correction output
                        outputs = outputs[:, :, f_dim:]
                        batch_y = batch_y[:, :, f_dim:]

                        pred = outputs[:batch_y.shape[0], :batch_y.shape[1], :]

                        if self.args.errcor_coef>0:
                            pred =  outputs_pred[:batch_y.shape[0], -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                        true = batch_y

                        opreds.append(pred)
                        otrues.append(true)

                        # Visual inspection
                        if i % 10 == 0:
                            input =  batch_x[:,:,:].detach().cpu().numpy()
                            oinput = oinput.detach().cpu().numpy()
                            if test_data.scale and self.args.inverse:
                                shape = input.shape
                                input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                                oinput = test_data.inverse_transform(oinput.reshape(shape[0] * shape[1], -1)).reshape(shape)

                            gts = []
                            pds = []
                            for ii in range(7):
                                gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                                gts.append(gt)
                                pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                                pds.append(pd)
                            visualm(gts, pds, os.path.join(folder_path, f"{data_pred_len}_ar{self.args.use_ar}-{ecm}-mr{self.args.errcor_coef}-{i}-{j}.pdf"))
                        oinput = vbatch_y
                        if i == 20000:
                            break
                    
                    # Merge multi-step outputs -> Inspect this code for evaluation from Hiep
                    opreds = np.concatenate(opreds, axis=1)
                    otrues = np.concatenate(otrues, axis=1)

                    opreds = opreds[:,:data_pred_len,:]
                    otrues = otrues[:,:data_pred_len,:]
                    
                    preds.append(opreds)
                    trues.append(otrues)    

            # -------------------------------------------------------------------------------------- Final Evaluation --------------------------------------------------------------------------------------
        
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './scratch/infer_results/' + setting + f'-' + ecm + f'-' + criterion_name + '/'

            mae, mse, rmse, mape, mspe, dtw = self._save_results(setting, folder_path, preds, trues)

            with open(f"{folder_path}/metrics-{data_pred_len}-{self.args.errcor_coef}.txt", 'w') as f:
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
                f.write('\n')
                f.write('\n')

            self.args.pred_len = self.ori_pred_len

        if not loaded_any_criterion:
            raise FileNotFoundError("No valid modelerr checkpoint found for any coeff and criterion.")
        return

    def train_infer_batch_with_trend_season(self, setting, ecm="linear", error_flags=None):
        # Adjust setting for predicted length
        setting, model_pred_len = self._resolve_backbone_setting(setting)
        print(setting)

        # Initialize wandb logging if enabled
        if self.args.wandb:
            run_id = f"{random.randint(1000, 9999)}-{setting}-{ecm}"
            wandb.init(
                project="timeecm",     
                name=run_id            
            )


        # Load test data
        test_data, test_loader = self._get_data(flag='val')
        train_data, _ = self._get_data(flag='train')

        total_len = len(test_data)
        train_len = int(0.7 * total_len)
        val_len = total_len - train_len

        # Split
        _, ecm_val_data = torch.utils.data.random_split(test_data, [train_len, val_len])
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=True)
        ecm_val_loader = torch.utils.data.DataLoader(ecm_val_data, batch_size=self.args.batch_size, shuffle=False)
        
        more_data_len = train_len

        if len(train_data) >= more_data_len:
            generator = torch.Generator().manual_seed(42)
            more_data, _ = torch.utils.data.random_split(train_data, [more_data_len, len(train_data) - more_data_len], generator=generator)
        else:
            raise ValueError(f"train_data too small ({len(train_data)}) to extract {val_len} samples.")
        
        ecm_val_data = torch.utils.data.ConcatDataset([ecm_val_data, more_data])

        ecm_val_loader = torch.utils.data.DataLoader(ecm_val_data, batch_size=self.args.batch_size, shuffle=False)



        data_pred_len =self.args.pred_len
        num_ar = math.ceil(data_pred_len/model_pred_len)
        self.args.pred_len = model_pred_len
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)


        # Load the trained backbone model
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        # Prediction, ground truth, and correction component containers
        preds = []
        trues = []
        folder_path = f'./scratch/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}/' + setting + f'-' + ecm +  '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        #-------------------------------------------------------------------------------------- Backbone Model Inference Phase --------------------------------------------------------------------------------------
        # Model inference with autoregressive prediction
        self.model.eval()
        correction_inputs = []
        correction_targets = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                obatch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                obatch_y_mark = batch_y_mark.float().to(self.device)

                for j in range(num_ar):
                    # Prepare decoder input and autoregressive input
                    batch_y = obatch_y[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    batch_y_mark = obatch_y_mark[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    if j==0:
                        enc_inp = batch_x
                        oinput = batch_x
                    else:
                        enc_inp =  pred_y
                       
                    batch_x = enc_inp 
                    dec_inp = self._make_dec_inp(batch_y)

                    # Forward pass
                    outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:,:,:]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    pred_y = outputs

                    # Handle inverse scaling if needed
                    if self.args.use_ar==0:
                        pred_y = obatch_y[:, self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]

                    batch_x_mark = batch_y_mark
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    vbatch_y = batch_y[:, :, f_dim:]
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    # Apply inverse transform
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
            
            
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs[:batch_y.shape[0], :batch_y.shape[1], :]
                    true = batch_y

                    if true.shape[1] < model_pred_len:
                        continue

                    # Construct seasonal and trend components
                    
                    torch_pred = torch.tensor(pred).to(self.device)
                    torch_true = torch.tensor(true).to(self.device)

                    pred_trend = moving_avg(torch_pred, kernel_size=self.args.kernel_size)
                    pred_seasonal = torch_pred - pred_trend
                    
                    true_trend    = moving_avg(torch_true, kernel_size=self.args.kernel_size)
                    true_seasonal = torch_true - true_trend
                    
                    pt = pred_trend[:, -self.args.pred_len:, :]         # predicted trends
                    ps = pred_seasonal[:, -self.args.pred_len:, :]         # predicted seasonal

                    
                    # Store correction data (input = [input, pred_trend, pred_seasonal], target = error_trend, error_seasonal)
                    corr_in = torch.cat([batch_x, pt, ps], dim=-1)         # => [B, T_enc_or_pred, C_total]
                    correction_inputs.append(corr_in)
                    trend_err    = true_trend[:, -self.args.pred_len:, :] - pt
                    seasonal_err = true_seasonal[:, -self.args.pred_len:, :] - ps
                    corr_tgt = torch.cat([trend_err, seasonal_err], dim=-1)
                    correction_targets.append(corr_tgt)
                    # print("correction inputs shape:", corr_in.shape, "correction targets shape:", corr_tgt.shape)
                
                    preds.append(pred)
                    trues.append(true)

                    # Visual inspection
                    if i % 10 == 0:
                        input =  batch_x[:,:,:].detach().cpu().numpy()
                        oinput = oinput.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                            oinput = test_data.inverse_transform(oinput.reshape(shape[0] * shape[1], -1)).reshape(shape)

                        gts = []
                        pds = []

                        for ii in range(min(7, true.shape[-1])):
                            gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                            gts.append(gt)
                            pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                            pds.append(pd)
                        visualm(gts, pds, os.path.join(folder_path, f"ar{self.args.use_ar}-{i}-{j}.pdf"))
                    oinput = vbatch_y
                    if i == 20000:
                        break

        # Concatenate predictions and ground truths
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = f'./scratch/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_{self.args.kernel_size}/' + setting + f'-' + ecm +  '/'

        self._save_results(setting, folder_path, preds, trues)

        # Prepare training data for error correction
        X = torch.cat(correction_inputs, dim=0)  # shape: (N, T, D)
        Y = torch.cat(correction_targets, dim=0)

        # Normalize inputs (pred) and targets (error)
        Y_mean = Y.mean(dim=(0, 1), keepdim=True)
        Y_std = Y.std(dim=(0, 1), keepdim=True) + 1e-6
        Y = (Y - Y_mean) / Y_std

        # Split dataset
        dataset = TensorDataset(X, Y)
        train_size = int(0.7 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size   # 20% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        input_dim = X.shape[-1]

        modelerr_full, optimizer_full, loss_fn_full, scheduler_full, is_torch_model = create_error_corrector(
                                                                ecm=ecm,
                                                                input_dim=input_dim,
                                                                T=X.shape[1],
                                                                output_dim=Y.shape[-1],
                                                                hidden_dim=self.args.err_h,
                                                                device=self.device,
                                                            )
            
        # -------------------------------------------------------------------------------------- Train Error Corrector Model --------------------------------------------------------------------------------------
        # Train the error model (Torch or Scikit-Learn)
        if is_torch_model:
        # -------------------------------------------------------------------------------------- Train Error Corrector Model (Torch) --------------------------------------------------------------------------------------
            best_val_loss = float('inf')  # Initialize average validation loss as infinity
            best_model_err_full = None

            for epoch in range(100):
                modelerr_full.train()

                total_loss = 0
                total_loss_trend = 0
                total_loss_seasonal = 0
                for xb, yb in train_loader:
                    optimizer_full.zero_grad()
                    # Train 1 full model for season and trend
                    pred_full = modelerr_full(xb)

                    pred_trend_full, pred_seasonal_full = pred_full.split(pred_full.shape[-1] // 2, dim=-1)
                    true_trend_full, true_seasonal_full = yb.split(pred_full.shape[-1] // 2, dim=-1)

                    
                    loss_trend = loss_fn_full(pred_trend_full, true_trend_full)
                    loss_seasonal = loss_fn_full(pred_seasonal_full, true_seasonal_full)

                    loss_full =  loss_seasonal * self.args.season_coef + loss_trend * self.args.trend_coef

                    loss_full.backward()
                    optimizer_full.step()
                    
                    total_loss += loss_full.item() * xb.size(0)
                    total_loss_trend += loss_trend.item() * xb.size(0)
                    total_loss_seasonal += loss_seasonal.item() * xb.size(0)


                avg_loss = total_loss / len(train_loader.dataset)
                avg_loss_trend = total_loss_trend / len(train_loader.dataset)
                avg_loss_seasonal = total_loss_seasonal / len(train_loader.dataset)

                # scheduler_trend.step(avg_loss_trend)
                # scheduler_season.step(avg_loss_seasonal)
                scheduler_full.step(avg_loss)

                print(f"Epoch {epoch+1} - Training Loss Trend: {avg_loss_trend:.4f} - Training Loss Seasonal: {avg_loss_seasonal:.4f}")
                if self.args.wandb:
                    wandb.log({
                        "train/total_loss": avg_loss,
                        "train/trend_loss": avg_loss_trend,
                        "train/seasonal_loss": avg_loss_seasonal,
                        "epoch": epoch + 1
                    })

                modelerr_full.eval()
                val_loss = 0
                val_loss_trend = 0
                val_loss_seasonal = 0

                with torch.no_grad():
                    for xb, yb in val_loader:

                        pred_full = modelerr_full(xb)
                        
                        # Split into trend and seasonal components to log the training loss
                        pred_trend, pred_season = pred_full.split(pred_full.shape[-1] // 2, dim=-1)
                        true_trend, true_season = yb.split(pred_full.shape[-1] // 2, dim=-1)

                        loss_trend = loss_fn_full(pred_trend, true_trend)
                        loss_seasonal = loss_fn_full(pred_season, true_season)

                        loss = loss_seasonal * self.args.season_coef + loss_trend * self.args.trend_coef

                        val_loss += loss.item() * xb.size(0)
                        val_loss_trend += loss_trend.item() * xb.size(0)
                        val_loss_seasonal += loss_seasonal.item() * xb.size(0)
                
                avg_val_loss = val_loss / len(val_loader.dataset)
                avg_loss_trend = val_loss_trend / len(val_loader.dataset)
                avg_loss_seasonal = val_loss_seasonal / len(val_loader.dataset)
                print(f"Epoch {epoch+1} - Validation Loss Trend: {avg_loss_trend:.4f} - Validation Loss Seasonal: {avg_loss_seasonal:.4f}")
                if self.args.wandb:
                    wandb.log({
                        "val/total_loss": avg_val_loss,
                        "val/trend_loss": avg_loss_trend,
                        "val/seasonal_loss": avg_loss_seasonal,
                        "epoch": epoch + 1
                    })

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_err_full = copy.deepcopy(modelerr_full)


        for criterion_name in ["MSE", "MAE"]:
            print("Searching for best error coefficient using criterion:", criterion_name)
            criterion = self._select_criterion(criterion_name=criterion_name)
            best_i = 0
            best_val_loss = float('inf')
            best_save_path = None

            print("Begin searching for best error coefficient")
            error_flags_to_use = [float(x) for x in error_flags] if error_flags else [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
            for i in error_flags_to_use:
                print(i)
                avg_val_loss = self.vali_with_seasonal_trend(ecm_val_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model_full=best_model_err_full)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_i = i
                    print(f"New best validation loss: {best_val_loss:.4f} with coef {i}.")

                    # Remove previous best checkpoint
                    if best_save_path is not None and os.path.exists(best_save_path):
                        os.remove(best_save_path)

                    # Save new best checkpoint
                    if is_torch_model:
                        best_save_path = os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-season-{self.args.season_coef}-trend-{self.args.trend_coef}-{criterion_name}-{self.args.kernel_size}.pth')
                        torch.save(best_model_err_full.state_dict(), best_save_path)
                    else:
                        best_save_path = os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-{criterion_name}.pkl')
                        joblib.dump(best_model_err_full, best_save_path)

            print(f"Best model for {criterion_name} saved to: {best_save_path}")


    

    def vali_with_seasonal_trend(self, vali_loader, criterion, error_coeff=0, is_torch_model=False, error_model_full=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = self._make_dec_inp(batch_y)
                outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if error_coeff == 0:
                    loss = criterion(pred, true)
                else:
                    pred_trend    = moving_avg(outputs, kernel_size=self.args.kernel_size)       # [B, T, C_total]
                    pred_seasonal = outputs - pred_trend                      # [B, T, C_total]

                    pt = pred_trend[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                    ps = pred_seasonal[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                    
                    meinput = torch.cat([batch_x, pt, ps], dim=-1)      

                    if is_torch_model:
                        perr_full = error_model_full(meinput)
                    else:
                        perr_full = torch.tensor(error_model_full.predict(meinput.cpu().numpy())).to(self.device)
                    perr_trend, perr_season = perr_full.split(perr_full.shape[-1] // 2, dim=-1)

                    perr = perr_trend + perr_season  # [B, pred_len, C]
                    if self.args.trend_coef == 0:
                        perr = perr_season
                    elif self.args.season_coef == 0:
                        perr = perr_trend

                    perr = perr.to(self.device)

                    outputs_pred = outputs + perr * error_coeff

                    loss = criterion(outputs_pred.to("cpu"), true)

                total_loss.append(loss.item())
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test_infer_batch_with_trend_season(self, setting, ecm="linear"):
        # Adjust model setting based on prediction length
        setting, model_pred_len = self._resolve_backbone_setting(setting)
        print(setting)

        # Load test data 
        test_data, test_loader = self._get_data(flag='test')
        data_pred_len =self.args.pred_len

        num_ar = math.ceil(data_pred_len/model_pred_len)
        self.args.pred_len = model_pred_len

        user_errcor_coef = self.args.errcor_coef

        # Load the backbone model
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))


        print('loading modelerr')

        modelerr_full, _, _, _, is_torch_model = create_error_corrector(ecm=ecm, input_dim=self.args.enc_in*3, T=model_pred_len, output_dim=self.args.enc_in*2, hidden_dim=self.args.err_h, device=self.device)
            
        # Automated process to find the best error correction coefficient that was recorded
        possible_coeffs = self._candidate_error_coeffs([0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1.0])
        model_loaded = False
        loaded_any_criterion = False
        for criterion_name in ["MSE", "MAE"]:
            # 1. RESET THE FLAG
            model_loaded = False

            # 2. ENSURE PRED_LEN IS CORRECT FOR THIS RUN
            self.args.pred_len = model_pred_len

            # 3. (Optional but safer) Clear GPU memory
            torch.cuda.empty_cache()
            for coeff in possible_coeffs:
                try:
                    modelerr_full.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-season-{self.args.season_coef}-trend-{self.args.trend_coef}-{criterion_name}-{self.args.kernel_size}.pth')))
                    modelerr_full.eval()
                    model_loaded = True
                    if user_errcor_coef == -1:
                        self.args.errcor_coef = coeff
                        print(f"Auto mode: using coefficient {coeff} from training checkpoint (criterion={criterion_name}).")
                    else:
                        self.args.errcor_coef = user_errcor_coef
                        print(f"Manual mode: using coefficient {user_errcor_coef} (checkpoint was trained with {coeff}).")

                    break
                except FileNotFoundError:
                    continue

            if not model_loaded:
                print("Current setting", setting, ecm, criterion_name)
                continue

            loaded_any_criterion = True

            preds = []
            trues = []
            folder_path = f'./scratch/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_{self.args.kernel_size}/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            self.model.eval()

            print("Using error correction coefficient:", self.args.errcor_coef)
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                    batch_x = batch_x.float().to(self.device)
                    obatch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    obatch_y_mark = batch_y_mark.float().to(self.device)
                    opreds = []
                    otrues = []
                    for j in range(num_ar):
                        batch_y = obatch_y[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                        batch_y_mark = obatch_y_mark[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                        if j==0:
                            enc_inp = batch_x
                            oinput = batch_x
                        else:
                            enc_inp =  pred_y
                        batch_x = enc_inp #Autoregression
                        dec_inp = self._make_dec_inp(batch_y)

                        # Forward pass through the model
                        outputs = self._backbone_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0

                        # Error correction step
                        if self.args.errcor_coef>0:
                            pred_trend    = moving_avg(outputs, kernel_size=self.args.kernel_size)       # [B, T, C_total]
                            pred_seasonal = outputs - pred_trend                      # [B, T, C_total]
                            pt = pred_trend[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                            ps = pred_seasonal[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                            meinput = torch.cat([batch_x, pt, ps], dim=-1)

                            if is_torch_model:
                                pred_full = modelerr_full(meinput)
                            else:
                                pred_full = torch.tensor(modelerr_full.predict(meinput.cpu().numpy())).to(self.device)
                            pred_trend, pred_season = pred_full.split(pred_full.shape[-1] // 2, dim=-1)
                            perr = pred_trend + pred_season  # [B, pred_len, C]

                            outputs_pred = outputs + perr * self.args.errcor_coef
                            outputs = outputs_pred

                        outputs = outputs[:, -self.args.pred_len:, :]

                        pred_y = outputs
                        if self.args.use_ar==0:
                            pred_y = obatch_y[:, self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]

                        batch_x_mark = batch_y_mark
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        vbatch_y = batch_y[:, :, f_dim:]
                        outputs = outputs.detach().cpu().numpy()
                        batch_y = batch_y.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = outputs.shape
                            outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                            batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                
                        outputs = outputs[:, :, f_dim:]
                        batch_y = batch_y[:, :, f_dim:]

                        pred = outputs[:batch_y.shape[0],:,:]
                        
                        # if self.args.errcor_coef>0:
                        #     pred =  outputs_pred[:batch_y.shape[0], -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                        true = batch_y

                        opreds.append(pred)
                        otrues.append(true)
                        if i % 10 == 0:
                            input =  batch_x[:,:,:].detach().cpu().numpy()
                            oinput = oinput.detach().cpu().numpy()
                            if test_data.scale and self.args.inverse:
                                shape = input.shape
                                input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                                oinput = test_data.inverse_transform(oinput.reshape(shape[0] * shape[1], -1)).reshape(shape)

                            gts = []
                            pds = []

                            # Visualization of predictions and ground truths
                            for ii in range(7):
                                gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                                gts.append(gt)
                                pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                                pds.append(pd)
                            visualm(gts, pds, os.path.join(folder_path, f"{data_pred_len}-ar{self.args.use_ar}-{ecm}-mr{self.args.errcor_coef}-{i}-{j}-seasonal{self.args.season_coef}-trend{self.args.trend_coef}.pdf"))
                        oinput = vbatch_y
                        if i == 20000:
                            break
                    opreds = np.concatenate(opreds, axis=1)
                    otrues = np.concatenate(otrues, axis=1)

                    opreds = opreds[:,:data_pred_len,:]
                    otrues = otrues[:,:data_pred_len,:]
                    
                    preds.append(opreds)
                    trues.append(otrues)        
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = f'./scratch/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_{self.args.kernel_size}/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
            print("Saving results to:", folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # dtw calculation
            if self.args.use_dtw:
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1,1)
                    y = trues[i].reshape(-1,1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = -999


            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            # if self.args.errcor_coef > 0:
            #     f = open(f"{folder_path}/metrics-{data_pred_len}-{coeff}.txt", 'w')
            # else:
            f = open(f"{folder_path}/metrics-{data_pred_len}-{self.args.errcor_coef}.txt", 'w')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            self.args.pred_len = self.ori_pred_len

        if not loaded_any_criterion:
            raise FileNotFoundError("No valid modelerr checkpoint found for any coeff and criterion.")
        return