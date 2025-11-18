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

from tqdm import tqdm
import wandb

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visualm
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from .utils_ECM import create_error_corrector, moving_avg, decompose_stl_1d


warnings.filterwarnings('ignore')

HOME_DIR = "/scratch/s223669184/project_data/Grant25/TimeSeriesECM"


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
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if error_coeff == 0:
                    loss = criterion(pred, true)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                    meinput = torch.cat([batch_x, outputs[:, -self.args.pred_len:, f_dim:]], dim=-1)

                    if is_torch_model:
                        perr = error_model(meinput)

                    else:
                        perr = error_model.predict(meinput.cpu().numpy())
                        perr = torch.tensor(perr).to(self.device)

                    outputs_pred = outputs + perr*error_coeff

                    loss = criterion(outputs_pred.to("cpu"), true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    
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
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = f'{HOME_DIR}/test_results/' + setting + '/'
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
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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
        folder_path = f'{HOME_DIR}/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(f"result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    
    def test_infer(self, setting, test=0, ecm="linear"):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = f'{HOME_DIR}/infer_results/' + setting + "-" + ecm + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        poses = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if i==0:
                    enc_inp = batch_x
                else:
                    enc_inp = torch.cat([enc_inp[:,1:,:], pred_y[:,:1,:]],  dim=1)

                    # enc_inp = torch.cat([enc_inptp_gt,enc_inptd], dim=2)
                obatch_x  = batch_x
                oinput = batch_x[:,:,:]
                if self.args.use_ar:
                    batch_x = enc_inp #Autoregression
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = outputs[:,:,:]
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                if i==0:
                    prev_outputs = outputs
                else:
                    outputs[:,:-1,:] = prev_outputs[:,1:,:]*self.args.alpha + outputs[:,:-1,:]*(1-self.args.alpha)
                    prev_outputs = outputs

                pred_y = outputs
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                poses.append(batch_x[0,-1, :].detach().cpu().numpy())
                if i % 1 == 0:
                    input =  batch_x[:,:,:].detach().cpu().numpy()
                    oinput = oinput.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        oinput = test_data.inverse_transform(oinput.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    gts = []
                    pds = []
                    gtps = []
                    pdps = []

                    for ii in range(7):
                        gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                        gts.append(gt)
                        pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                        pds.append(pd)
                        gtps.append(obatch_x[0,:,ii].detach().cpu().numpy())
                        pdps.append(batch_x[0,:,ii].detach().cpu().numpy())
                    visualm(gts, pds, os.path.join(folder_path, str(i) + '.pdf'))
                if i == 200:
                    break
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'{HOME_DIR}/infer_results/' + setting + '-' + ecm + '/'
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
        f = open(f"result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

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
        # model_pred_len = self.args.seq_len
        # setting_components = setting.split("_")
        # print(setting_components)
        # setting_components[5] = str(model_pred_len)
        # setting_components[11] = "pl"+str(model_pred_len)
        # setting = "_".join(setting_components)
        # print(setting)

        # # Load test data
        # test_data, test_loader = self._get_data(flag='test')
        # data_pred_len =self.args.pred_len
        # num_ar = math.ceil(data_pred_len/model_pred_len)

        # # Update model prediction length to match training time
        # self.args.pred_len = model_pred_len

        # Load backbone model
        # self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        # print('loading model')
        # self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))



        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
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
        num_ar = math.ceil(data_pred_len//model_pred_len)
        print("Num_ar", num_ar)

        # Override pred_len for backbone model prediction (typically to 96)
        self.args.pred_len = model_pred_len

        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        print(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth'))

        # Load backbone model
        self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))

        # -------------------------------------------------------------------------------------- Inference Phase with Backbone Model --------------------------------------------------------------------------------------
        preds = []
        trues = []
        folder_path = f'{HOME_DIR}/infer_results/' + setting + f'-' + ecm +  '/'
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
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # Model prediction
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
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

                    pred = outputs[:batch_y.shape[0],:,:]
                    true = batch_y
                    
                    # Store correction data (input = [input, pred], target = error)
                    correction_inputs.append(torch.cat([torch.tensor(batch_x),torch.tensor(pred_y[:, -self.args.pred_len:, f_dim:]).to(self.device)],dim=-1))
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
                        gtps = []
                        pdps = []

                        for ii in range(7):
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
        folder_path = f'{HOME_DIR}/infer_results/' + setting + f'-' + ecm +  '/'
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
        f = open(f"result_long_term_forecast.txt", 'w')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # -------------------------------------------------------------------------------------- Prepare Correction Dataset --------------------------------------------------------------------------------------
        X = torch.cat(correction_inputs, dim=0)  # shape: (N, T, D)
        Y = torch.cat(correction_targets, dim=0)
        print(X.shape)
        print(Y.shape)
        from torch.utils.data import TensorDataset, DataLoader, random_split
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
            best_val_loss_MAE = float('inf')  # Initialize best validation loss as infinity
            best_val_loss_MSE = float('inf')  # Initialize best validation loss as infinity
            best_model_err_MAE = None
            best_model_err_MSE = None
            best_model_err = None   
            best_val_loss = float('inf')  # Initialize best validation loss as infinity
            best_i_MAE = 0
            best_i_MSE = 0

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
                MSE_loss_total = 0
                MAE_loss_total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = modelerr(xb)
                        loss = loss_fn(pred, yb)

                        val_loss += loss.item() * xb.size(0)
                        if self.args.is_training == 7:
                            MSE_loss = torch.nn.functional.mse_loss(pred, yb)
                            MAE_loss = torch.nn.functional.l1_loss(pred, yb)
                            MSE_loss_total += MSE_loss.item() * xb.size(0)  
                            MAE_loss_total += MAE_loss.item() * xb.size(0)
                
                avg_val_loss = val_loss / len(val_loader.dataset)
                print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
                if self.args.is_training == 7:
                    print(f"Epoch {epoch+1} - Validation MSE Loss: {MSE_loss_total/len(val_loader.dataset):.4f}")
                    print(f"Epoch {epoch+1} - Validation MAE Loss: {MAE_loss_total/len(val_loader.dataset):.4f}")

                # Check if the current validation loss is the best we've seen so far
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # print(f"New best validation loss: {best_val_loss:.4f}.")
                    best_model_err = modelerr
                
                # ---------NEW----------------------------
            #     modelerr.eval()
            #     for criterion_name in ["MAE", "MSE"]:
            #         # print("Searching for best error coefficient using criterion:", criterion_name)
            #         criterion = self._select_criterion(criterion_name=criterion_name)
            #         # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

            #         print("Begin searching for best error coefficient")
            #         for i in error_flags:
            #             avg_val_loss = self.vali(ecm_val_data, ecm_val_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model=modelerr)
            #             if criterion_name == "MAE":
            #                 if avg_val_loss < best_val_loss_MAE:
            #                     best_val_loss_MAE = avg_val_loss
            #                     best_i_MAE = i
            #                     best_model_err_MAE = modelerr   
            #                     print(f"New best validation loss: {best_val_loss_MAE:.4f} with coef {i}.")
            #             elif criterion_name == "MSE":
            #                 if avg_val_loss < best_val_loss_MSE:
            #                     best_val_loss_MSE = avg_val_loss
            #                     best_i_MSE = i
            #                     best_model_err_MSE = modelerr
            #                     print(f"New best validation loss: {best_val_loss_MSE:.4f} with coef {i}.")
            # print("Training complete.")
                # ---------NEW----------------------------

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

            # Validation
            # val_X, val_Y = [], []
            # for xb, yb in val_loader:
            #     val_X.append(xb.cpu().numpy())
            #     val_Y.append(yb.cpu().numpy())
            # val_X = np.concatenate(val_X, axis=0)
            # val_Y = np.concatenate(val_Y, axis=0)

            # val_pred = modelerr.predict(val_X)
            # val_loss = np.mean(np.abs(val_pred - val_Y))  # L1 loss
            # print(f"Validation Loss: {val_loss:.4f}")
            best_model_err = modelerr

            # ---------NEW----------------------------
            # for criterion_name in ["MAE", "MSE"]:
            #     # print("Searching for best error coefficient using criterion:", criterion_name)
            #     criterion = self._select_criterion(criterion_name=criterion_name)
            #     # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

            #     print("Begin searching for best error coefficient")
            #     for i in error_flags:
            #         avg_val_loss = self.vali(ecm_val_data, ecm_val_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model=modelerr)
            #         if criterion_name == "MAE":
            #             if avg_val_loss < best_val_loss_MAE:
            #                 best_val_loss_MAE = avg_val_loss
            #                 best_i_MAE = i
            #                 best_model_err_MAE = modelerr   
            #                 print(f"New best validation loss: {best_val_loss_MAE:.4f} with coef {i}.")
            #         elif criterion_name == "MSE":
            #             if avg_val_loss < best_val_loss_MSE:
            #                 best_val_loss_MSE = avg_val_loss
            #                 best_i_MSE = i
            #                 best_model_err_MSE = modelerr
            #                 print(f"New best validation loss: {best_val_loss_MSE:.4f} with coef {i}.")
            # print("Training complete.")

        if self.args.is_training == 7:
            print("Finish training error corrector model.")
            print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
            print(f"Epoch {epoch+1} - Validation MSE Loss: {MSE_loss_total/len(val_loader.dataset):.4f}")
            print(f"Epoch {epoch+1} - Validation MAE Loss: {MAE_loss_total/len(val_loader.dataset):.4f}")
            return

        for criterion_name in ["MAE", "MSE"]:
            print("Searching for best error coefficient using criterion:", criterion_name)
            criterion = self._select_criterion(criterion_name=criterion_name)
            # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

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
                torch.save(best_model_err.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-{criterion_name}.pth'))
            else:
                save_path = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-{criterion_name}.pkl')
                joblib.dump(best_model_err, save_path)
            
        # if is_torch_model:
        #     torch.save(best_model_err_MAE.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i_MAE}-MAE.pth'))
        #     torch.save(best_model_err_MSE.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i_MSE}-MSE.pth'))
        # else:
        #     save_path = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i_MAE}-MAE.pkl')
        #     joblib.dump(best_model_err_MAE, save_path)
        #     save_path = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i_MSE}-MSE.pkl')
        #     joblib.dump(best_model_err_MSE, save_path)



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
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
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
        self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))
        # -------------------------------------------------------------------------------------- Load trained error corrector model --------------------------------------------------------------------------------------

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
        possible_coeffs = [0, 0.1, 0.3, 0.5, 0.7, 1]
        model_loaded = False
        for criterion_name in ["MAE", "MSE"]:
            for coeff in possible_coeffs:
                try:
                    if is_torch_model:
                        print("Here1")
                        print(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pth'))
                        modelerr.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pth')))
                        modelerr.eval()  # Set to evaluation mode
                        model_loaded = True
                        print("Modelerr loaded successfully.")
                        break
                    else:
                        print("Here1")
                        print(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pth'))
                        load_path = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pkl')
                        modelerr = joblib.load(load_path)
                        model_loaded = True 
                        print("Modelerr loaded successfully.")
                        break
                except FileNotFoundError:
                    continue
            if not model_loaded:
                raise FileNotFoundError("No valid modelerr checkpoint found for any coeff and criterion.")

            # -------------------------------------------------------------------------------------- Begin inference --------------------------------------------------------------------------------------
            preds = []
            trues = []
            folder_path = f'{HOME_DIR}/infer_results/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
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
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        
                        # Run prediction
                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
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

                        pred = outputs[:batch_y.shape[0],:,:]

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
            folder_path = f'{HOME_DIR}/infer_results/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
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
        return

    def train_infer_batch_with_trend_season(self, setting, ecm="linear", error_flags=None):
        # Adjust setting for predicted length
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
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
        num_ar = math.ceil(data_pred_len//model_pred_len)
        self.args.pred_len = model_pred_len
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)


        # Load the trained backbone model
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))

        # Prediction, ground truth, and correction component containers
        preds = []
        trues = []
        folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}/' + setting + f'-' + ecm +  '/'

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
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    # Forward pass
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
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

                    pred = outputs[:batch_y.shape[0],:,:]
                    true = batch_y

                    # Construct seasonal and trend components
                    
                    torch_pred = torch.tensor(pred).to(self.device)
                    torch_true = torch.tensor(true).to(self.device)

                    pred_trend = moving_avg(torch_pred, kernel_size=self.args.kernel_size)
                    pred_seasonal = torch_pred - pred_trend
                    
                    true_trend    = moving_avg(torch_true, kernel_size=self.args.kernel_size)
                    true_seasonal = torch_true - true_trend
                    
                    pt = pred_trend[:, -self.args.pred_len:, :]         # predicted trends
                    ps = pred_seasonal[:, -self.args.pred_len:, :]         # predicted seasonal

                    # print("Pred shape:", pred.shape, "True shape:", true.shape, 
                    #       "Predicted Trend shape:", pt.shape, "Predicted Seasonal shape:", ps.shape)
                    # print("Batch X shape:", batch_x.shape, "Batch Y shape:", batch_y.shape)
                    
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

                        for ii in range(7):
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
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}/' + setting + f'-' + ecm +  '/'
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
            
        # Save metrics and prediction
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(f"result_long_term_forecast.txt", 'w')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Prepare training data for error correction
        X = torch.cat(correction_inputs, dim=0)  # shape: (N, T, D)
        Y = torch.cat(correction_targets, dim=0)

        from torch.utils.data import TensorDataset, DataLoader, random_split
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


        if ecm == "random_forest" or ecm == "logistic" or ecm == "xgboost":
            input_dim=X.shape[-1] 

            modelerr_full, _, _, _, is_torch_model = create_error_corrector(
                                                        ecm=ecm,
                                                        input_dim=input_dim,
                                                        T=X.shape[1],
                                                        output_dim=Y.shape[-1],
                                                        hidden_dim=self.args.err_h,
                                                        device=self.device,
                                                    )

        else:
            input_dim=X.shape[-1] 

            modelerr_full, optimizer_full, loss_fn_full, scheduler_full, is_torch_model = create_error_corrector(
                                                                    ecm=ecm,
                                                                    input_dim=input_dim,
                                                                    T=X.shape[1],
                                                                    output_dim=Y.shape[-1],
                                                                    hidden_dim=self.args.err_h,
                                                                    device=self.device,
                                                                )
            

            # input_dim=X.shape[-1] // 3 * 2

            # modelerr_season, optimizer_season, loss_fn_season, scheduler_season, is_torch_model = create_error_corrector(
            #                                                         ecm=ecm,
            #                                                         input_dim=input_dim,
            #                                                         T=X.shape[1],
            #                                                         output_dim=Y.shape[-1] // 2,
            #                                                         hidden_dim=self.args.err_h,
            #                                                         device=self.device
            #                                                     )
            
            # modelerr_trend, optimizer_trend, loss_fn_trend, scheduler_trend, is_torch_model = create_error_corrector(
            #                                                 ecm=ecm,
            #                                                 input_dim=input_dim,
            #                                                 T=X.shape[1],
            #                                                 output_dim=Y.shape[-1] // 2,
            #                                                 hidden_dim=self.args.err_h,
            #                                                 device=self.device
            #                                             )
        # -------------------------------------------------------------------------------------- Train Error Corrector Model --------------------------------------------------------------------------------------
        # Train the error model (Torch or Scikit-Learn)
        if is_torch_model:
        # -------------------------------------------------------------------------------------- Train Error Corrector Model (Torch) --------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
            # best_val_loss_trend = float('inf')  # Initialize best validation loss as infinity
            # best_val_loss_seasonal = float('inf')  # Initialize best validation loss as infinity
            # best_model_err_trend = None
            # best_model_err_season = None

            best_val_loss = float('inf')  # Initialize average validation loss as infinity
            best_model_err_full = None

            for epoch in range(100):
                modelerr_full.train()
                # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                # modelerr_season.train()
                # modelerr_trend.train()

                total_loss = 0
                total_loss_trend = 0
                total_loss_seasonal = 0
                for xb, yb in train_loader:
                    optimizer_full.zero_grad()
                    # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                    # optimizer_season.zero_grad()
                    # optimizer_trend.zero_grad()


                    # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                    # B, T, C_x = xb.shape
                    # _, _, C_y2 = yb.shape       # 14
                    # C_err = C_y2 // 2           # 7
                    # C_hist = C_x - C_y2         # 21 - 14 = 7
                    # xb_trend =  xb[:, :, :C_hist + C_err]
                    # yb_trend = yb[:, :, :C_err] 
                    
                    # xb_season = torch.cat([xb[:, :, :C_hist], xb[:, :, C_hist+C_err:C_hist+2*C_err]], dim=-1)
                    # yb_season = yb[:, :, C_err:]                  # shape (64, 96,  7)

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

                # Validation phase

                # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                # modelerr_trend.eval()
                # modelerr_season.eval()

                modelerr_full.eval()
                val_loss = 0
                val_loss_trend = 0
                val_loss_seasonal = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                        # B, T, C_x = xb.shape
                        # _, _, C_y2 = yb.shape       # 14
                        # C_err = C_y2 // 2           # 7
                        # C_hist = C_x - C_y2         # 21 - 14 = 7
                        # xb_trend =  xb[:, :, :C_hist + C_err]
                        # yb_trend = yb[:, :, :C_err] 
                        
                        # xb_season = torch.cat([xb[:, :, :C_hist], xb[:, :, C_hist+C_err:C_hist+2*C_err]], dim=-1)
                        # yb_season = yb[:, :, C_err:]                  # shape (64, 96,  7)
        
                        # pred_trend = modelerr_trend(xb_trend)
                        # pred_season = modelerr_season(xb_season)

                        # loss_trend = loss_fn_trend(pred_trend, yb_trend)
                        # loss_seasonal = loss_fn_season(pred_season, yb_season)


                        pred_full = modelerr_full(xb)
                        
                        # Split into trend and seasonal components to log the training loss
                        pred_trend, pred_season = pred_full.split(pred_full.shape[-1] // 2, dim=-1)
                        true_trend, true_season = yb.split(pred_full.shape[-1] // 2, dim=-1)

                        loss_trend = loss_fn_full(pred_trend, true_trend)
                        loss_seasonal = loss_fn_full(pred_season, true_season)

                        loss = loss_trend * self.args.season_coef + loss_seasonal * self.args.trend_coef

                        val_loss += loss.item() * xb.size(0)
                        val_loss_trend += loss_trend.item() * xb.size(0)
                        val_loss_seasonal += loss_seasonal.item() * xb.size(0)

                        # val_loss = val_loss_trend * self.args.trend_coef + val_loss_seasonal * self.args.season_coef
                
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
                    best_model_err_full = modelerr_full
                    # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                    # best_model_err_season = modelerr_season
                    # best_model_err_trend = modelerr_trend

        else:
            # -------------------------------------------------------------------------------------- Train Error Corrector Model (Scikit-Learn) --------------------------------------------------------------------------------------
            # Collect all training data

            # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
            # train_X_trend, train_Y_trend, train_X_seasonal, train_Y_seasonal = [], [], [], []
            train_X, train_Y = [], []
            for xb, yb in train_loader:
                # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
                # B, T, C_x = xb.shape
                # _, _, C_y2 = yb.shape       # 14
                # C_err = C_y2 // 2           # 7
                # C_hist = C_x - C_y2         # 21 - 14 = 7

                # xb_trend =  xb[:, :, :C_hist + C_err]
                # yb_trend = yb[:, :, :C_err] 
                
                # xb_season = torch.cat([xb[:, :, :C_hist], xb[:, :, C_hist+C_err:C_hist+2*C_err]], dim=-1)
                # yb_season = yb[:, :, C_err:]                  # shape (64, 96,  7)

                # train_X_trend.append(xb_trend.cpu().numpy())
                # train_Y_trend.append(yb_trend.cpu().numpy())

                # train_X_seasonal.append(xb_season.cpu().numpy())
                # train_Y_seasonal.append(yb_season.cpu().numpy())    
                
                train_X.append(xb.cpu().numpy())
                train_Y.append(yb.cpu().numpy())

            # ----------------------------------------------------------------------------Uncomment if used separate models for trend and season
            # train_X_trend = np.concatenate(train_X_trend, axis=0)
            # train_Y_trend = np.concatenate(train_Y_trend, axis=0)
            # train_X_seasonal = np.concatenate(train_X_seasonal, axis=0)
            # train_Y_seasonal = np.concatenate(train_Y_seasonal, axis=0)
            # modelerr_season.fit(train_X_seasonal, train_Y_seasonal)
            # modelerr_trend.fit(train_X_trend, train_Y_trend)

            train_X = np.concatenate(train_X, axis=0)
            train_Y = np.concatenate(train_Y, axis=0)

            # Fit the model
            modelerr_full.fit(train_X, train_Y)        

            # Validation
            # val_X, val_Y = [], []
            # for xb, yb in val_loader:
            #     val_X.append(xb.cpu().numpy())
            #     val_Y.append(yb.cpu().numpy())
            # val_X = np.concatenate(val_X, axis=0)
            # val_Y = np.concatenate(val_Y, axis=0)

            # val_pred = modelerr_full.predict(val_X)
            # val_loss = np.mean(np.abs(val_pred - val_Y))  # L1 loss
            # print(f"Validation Loss: {val_loss:.4f}")
            # best_model_err_trend = modelerr_trend
            # best_model_err_season = modelerr_season
            best_model_err_full = modelerr_full

        for criterion_name in ["MSE","MAE"]:       
            print("Searching for best error coefficient using criterion:", criterion_name)
            criterion = self._select_criterion(criterion_name=criterion_name)
            # vali_data, vali_loader = self._get_data(flag='val')
            best_i = 0
            best_val_loss = float('inf')  # Initialize best validation loss as infinity

            # Hyperparameter search for error correction weight
            print("Begin searching for best error coefficient")
            for i in error_flags:
                print(i)
                # ----------------------------------------------------------------------------- Uncomment if used separate models for trend and season
                # avg_val_loss = self.vali_with_seasonal_trend(vali_data, vali_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model_seasonal=best_model_err_season, error_model_trend=best_model_err_trend, error_model_full=best_model_err_full)
                avg_val_loss = self.vali_with_seasonal_trend(ecm_val_data, ecm_val_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model_full=best_model_err_full)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_i = i
                    print(f"New best validation loss: {best_val_loss:.4f} with coef {i}.")

            # Save best model
            if is_torch_model:
                torch.save(best_model_err_full.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-season-{self.args.season_coef}-trend-{self.args.trend_coef}-{criterion_name}.pth'))
                # Uncomment if used separate models for trend and season
                # torch.save(best_model_err_season.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-season-{self.args.season_coef}.pth'))
                # torch.save(best_model_err_trend.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-trend-{self.args.trend_coef}.pth'))        
            else:
                save_path = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-{criterion_name}.pkl')
                # Uncomment if used separate models for trend and season
                # save_path_season = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-season.pkl')
                # save_path_trend = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-trend.pkl')

                joblib.dump(best_model_err_full, save_path)


    

    def vali_with_seasonal_trend(self, vali_data, vali_loader, criterion, error_coeff=0, is_torch_model=False, error_model_seasonal=None, error_model_trend=None, error_model_full=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if error_coeff == 0:
                    loss = criterion(pred, true)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                    pred_trend    = moving_avg(outputs, kernel_size=self.args.kernel_size)       # [B, T, C_total]
                    pred_seasonal = outputs - pred_trend                      # [B, T, C_total]

                    pt = pred_trend[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                    ps = pred_seasonal[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]

                    # ------------------------------------------------------------------------------------Uncomment if used separate models for trend and season
                    # meinput_trend = torch.cat([
                    #     batch_x,    # [B, pred_len, C_history]
                    #     pt,         # [B, pred_len, C]
                    # ], dim=-1)      # → [B, pred_len, C_history + 2*C]

                    # meinput_season = torch.cat([
                    #     batch_x,    # [B, pred_len, C_history]
                    #     ps,         # [B, pred_len, C]
                    # ], dim=-1)      # → [B, pred_len, C_history + 2*C]
                    
                    meinput = torch.cat([batch_x, pt, ps], dim=-1)      

                    if is_torch_model:
                        # -------------------------------------------------------------------------------Uncomment if used separate models for trend and season
                        # perr_trend = error_model_trend(meinput_trend)
                        # perr_season = error_model_seasonal(meinput_season)
                        perr_full = error_model_full(meinput)
                        perr_trend, perr_season = perr_full.split(perr_full.shape[-1] // 2, dim=-1)

                    else:
                        perr = error_model_full.predict(meinput.cpu().numpy())
                        perr_trend, perr_season = perr.split(perr.shape[-1] // 2, axis=-1)

                        # -------------------------------------------------------------------------------Uncomment if used separate models for trend and season
                        # perr_trend = error_model_trend.predict(meinput_trend.cpu().numpy())
                        # perr_season = error_model_seasonal.predict(meinput_season.cpu().numpy())

                        # perr_trend = torch.tensor(perr_trend).to(self.device)
                        # perr_season = torch.tensor(perr_season).to(self.device)


                    perr = perr_trend + perr_season  # [B, pred_len, C]
                    
                    if self.args.trend_coef == 0:
                        perr = perr_season
                    elif self.args.season_coef == 0:
                        perr = perr_trend
                    
                    perr = torch.tensor(perr).to(self.device)

                    outputs_pred = outputs + perr * error_coeff

                    loss = criterion(outputs_pred.to("cpu"), true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test_infer_batch_with_trend_season(self, setting, ecm="linear"):
        # Adjust model setting based on prediction length
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
        print(setting)

        # Load test data 
        test_data, test_loader = self._get_data(flag='test')
        data_pred_len =self.args.pred_len

        num_ar = math.ceil(data_pred_len/model_pred_len)
        self.args.pred_len = model_pred_len

        # Load the backbone model
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))
        

        input_dim=self.args.enc_in*2    # input_dim for error corrector model (input + trend + seasonal)
        out_dim = self.args.enc_in*1    # output_dim for error corrector model
        print('loading modelerr')

        if ecm == "random_forest" or ecm == "logistic" or ecm == "xgboost":
            modelerr, _, _, _, is_torch_model = create_error_corrector(ecm=ecm, input_dim=input_dim, T=model_pred_len, output_dim=out_dim, hidden_dim=self.args.err_h, device=self.device)
        else:
            modelerr_full, _, _, _, is_torch_model = create_error_corrector(ecm=ecm, input_dim=self.args.enc_in*3, T=model_pred_len, output_dim=self.args.enc_in*2, hidden_dim=self.args.err_h, device=self.device)
            
            # -------------------------------------------------------------- Uncomment if used separate models for trend and season
            # modelerr_trend, _, _, _, is_torch_model = create_error_corrector(ecm=ecm, input_dim=input_dim, T=model_pred_len, output_dim=out_dim, hidden_dim=self.args.err_h, device=self.device)
            # modelerr_season, _, _, _, is_torch_model = create_error_corrector(ecm=ecm, input_dim=input_dim, T=model_pred_len, output_dim=out_dim, hidden_dim=self.args.err_h, device=self.device)

        
        # Automated process to find the best error correction coefficient that was recorded
        possible_coeffs = [0, 0.1, 0.3, 0.5, 0.7, 1] 
        model_loaded = False
        for criterion_name in ["MSE", "MAE"]:
            for coeff in possible_coeffs:
                try:
                    if is_torch_model:
                        modelerr_full.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-season-{self.args.season_coef}-trend-{self.args.trend_coef}-{criterion_name}.pth')))
                        modelerr_full.eval()
                        model_loaded = True 
                        break
                        # -------------------------------------------------------------- Uncomment if used separate models for trend and season    
                        # modelerr_trend.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-trend-{self.args.trend_coef}.pth')))
                        # modelerr_season.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-season-{self.args.season_coef}.pth'))) 
                        # modelerr_trend.eval()  
                        # modelerr_season.eval()
                    else:
                        # load_path_trend = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-trend.pkl')
                        # load_path_season = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-season.pkl')
                        load_path = os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-{criterion_name}.pkl')

                        # modelerr_trend = joblib.load(load_path_trend)
                        # modelerr_season = joblib.load(load_path_season)
                        modelerr = joblib.load(load_path)
                        model_loaded = True
                        break
                except FileNotFoundError:
                    continue

            if not model_loaded:
                raise FileNotFoundError("No valid modelerr checkpoint found for any coeff and criterion.")

            preds = []
            trues = []
            folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
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
                        if j==0:
                            enc_inp = batch_x
                            oinput = batch_x
                        else:
                            enc_inp =  pred_y
                        batch_x = enc_inp #Autoregression
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        
                        # Forward pass through the model
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0

                        # Error correction step
                        if self.args.errcor_coef>0:
                            pred_trend    = moving_avg(outputs, kernel_size=self.args.kernel_size)       # [B, T, C_total]
                            pred_seasonal = outputs - pred_trend                      # [B, T, C_total]
                            pt = pred_trend[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                            ps = pred_seasonal[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                            meinput_trend = torch.cat([batch_x, pt], dim=-1)      
                            meinput_season = torch.cat([batch_x, ps], dim=-1)      
                            meinput = torch.cat([batch_x, pt, ps], dim=-1)      

                            if is_torch_model:
                                # -------------------------------------------------------------------------------Uncomment if used separate models for trend and season
                                # perr_trend = modelerr_trend(meinput_trend)
                                # perr_season = modelerr_trend(meinput_season)
                                # perr = perr_trend + perr_season  # [B, pred_len, C]

                                pred_full = modelerr_full(meinput)
                                pred_trend, pred_season = pred_full.split(pred_full.shape[-1] // 2, dim=-1)
                                perr = pred_trend + pred_season  # [B, pred_len, C]
                                # if self.args.trend_coef == 0:
                                #     perr = pred_season
                                # elif self.args.season_coef == 0:
                                #     perr = pred_trend
                            else:
                                # -------------------------------------------------------------------------------Uncomment if used separate models for trend and season
                                # perr_trend = modelerr_trend.predict(meinput_trend.cpu().numpy())
                                # perr_season = modelerr_season.predict(meinput_season.cpu().numpy())
                                # perr_trend = torch.tensor(perr_trend).to(self.device)
                                # perr_season = torch.tensor(perr_season).to(self.device)

                                perr = modelerr.predict(meinput.cpu().numpy())
                                perr = torch.tensor(perr).to(self.device)


                            # trend_err_pred, seasonal_err_pred = perr.chunk(2, dim=-1)
                            # perr = (perr_trend * self.args.trend_coef + perr_season * self.args.season_coef) * 2 # [B, pred_len, C]
                            outputs_pred = outputs + perr*self.args.errcor_coef


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
                        
                        if self.args.errcor_coef>0:
                            pred =  outputs_pred[:batch_y.shape[0], -self.args.pred_len:, f_dim:].detach().cpu().numpy()
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
            folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
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
        return







    def train_infer_batch_with_trend_season_ablation1(self, setting, ecm="linear", error_flags=None):
        # Adjust setting for predicted length
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
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
        num_ar = math.ceil(data_pred_len//model_pred_len)
        self.args.pred_len = model_pred_len
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)


        # Load the trained backbone model
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))

        # Prediction, ground truth, and correction component containers
        preds = []
        trues = []
        folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_ablation1/' + setting + f'-' + ecm +  '/'

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
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    # Forward pass
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
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

                    pred = outputs[:batch_y.shape[0],:,:]
                    true = batch_y

                    # Construct seasonal and trend components
                    
                    torch_pred = torch.tensor(pred).to(self.device)
                    torch_true = torch.tensor(true).to(self.device)

                    pred_trend = moving_avg(torch_pred, kernel_size=self.args.kernel_size)
                    pred_seasonal = torch_pred - pred_trend                    
                    
                    pt = pred_trend[:, -self.args.pred_len:, :]         # predicted trends
                    ps = pred_seasonal[:, -self.args.pred_len:, :]         # predicted seasonal

                    # Store correction data (input = [input, pred_trend, pred_seasonal], target = error_trend, error_seasonal)
                    corr_in = torch.cat([batch_x, pt, ps], dim=-1)         # => [B, T_enc_or_pred, C_total]
                    correction_inputs.append(corr_in)
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

                        for ii in range(7):
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
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_ablation1/' + setting + f'-' + ecm +  '/'
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
            
        # Save metrics and prediction
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open(f"result_long_term_forecast.txt", 'w')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Prepare training data for error correction
        X = torch.cat(correction_inputs, dim=0)  # shape: (N, T, D)
        Y = torch.cat(correction_targets, dim=0)

        from torch.utils.data import TensorDataset, DataLoader, random_split
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



        input_dim=X.shape[-1] 

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

                    pred_full = modelerr_full(xb)
                    
                    loss_full = loss_fn_full(pred_full, yb)

                    loss_full.backward()
                    optimizer_full.step()
                    
                    total_loss += loss_full.item() * xb.size(0)

                avg_loss = total_loss / len(train_loader.dataset)
                avg_loss_trend = total_loss_trend / len(train_loader.dataset)
                avg_loss_seasonal = total_loss_seasonal / len(train_loader.dataset)

                scheduler_full.step(avg_loss)

                print(f"Epoch {epoch+1} - Training Loss Trend: {avg_loss_trend:.4f} - Training Loss Seasonal: {avg_loss_seasonal:.4f}")
                if self.args.wandb:
                    wandb.log({
                        "train/total_loss": avg_loss,
                        "train/trend_loss": avg_loss_trend,
                        "train/seasonal_loss": avg_loss_seasonal,
                        "epoch": epoch + 1
                    })

                # Validation phase

                modelerr_full.eval()
                val_loss = 0
                val_loss_trend = 0
                val_loss_seasonal = 0

                with torch.no_grad():
                    for xb, yb in val_loader:

                        pred_full = modelerr_full(xb)
                        
                        loss = loss_fn_full(yb, pred_full)


                        val_loss += loss.item() * xb.size(0)
                
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
                    best_model_err_full = modelerr_full


        for criterion_name in ["MSE","MAE"]:       
            print("Searching for best error coefficient using criterion:", criterion_name)
            criterion = self._select_criterion(criterion_name=criterion_name)
            # vali_data, vali_loader = self._get_data(flag='val')
            best_i = 0
            best_val_loss = float('inf')  # Initialize best validation loss as infinity

            # Hyperparameter search for error correction weight
            print("Begin searching for best error coefficient")
            for i in error_flags:
                print(i)
                avg_val_loss = self.vali_with_seasonal_trend_ablation1(ecm_val_data, ecm_val_loader, criterion, error_coeff=i, is_torch_model=is_torch_model, error_model_full=best_model_err_full)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_i = i
                    print(f"New best validation loss: {best_val_loss:.4f} with coef {i}.")

            # Save best model
            if is_torch_model:
                torch.save(best_model_err_full.state_dict(), os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{best_i}-season-{self.args.season_coef}-trend-{self.args.trend_coef}-{criterion_name}-ablation1.pth'))


    def vali_with_seasonal_trend_ablation1(self, vali_data, vali_loader, criterion, error_coeff=0, is_torch_model=False, error_model_seasonal=None, error_model_trend=None, error_model_full=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if error_coeff == 0:
                    loss = criterion(pred, true)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                    pred_trend    = moving_avg(outputs, kernel_size=self.args.kernel_size)       # [B, T, C_total]
                    pred_seasonal = outputs - pred_trend                      # [B, T, C_total]

                    pt = pred_trend[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]
                    ps = pred_seasonal[:, -self.args.pred_len:, f_dim:]       # [B, pred_len, C]

                    
                    meinput = torch.cat([batch_x, pt, ps], dim=-1)      

                    if is_torch_model:
                        perr_full = error_model_full(meinput)

                    perr = perr_full

                    perr = torch.tensor(perr).to(self.device)

                    outputs_pred = outputs + perr * error_coeff

                    loss = criterion(outputs_pred.to("cpu"), true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test_infer_batch_with_trend_season_ablation1(self, setting, ecm="linear"):
        # Adjust model setting based on prediction length
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
        print(setting)

        # Load test data 
        test_data, test_loader = self._get_data(flag='test')
        data_pred_len =self.args.pred_len

        num_ar = math.ceil(data_pred_len/model_pred_len)
        self.args.pred_len = model_pred_len

        # Load the backbone model
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, 'checkpoint.pth')))
        
        print('loading modelerr')


        modelerr_full, _, _, _, is_torch_model = create_error_corrector(ecm=ecm, input_dim=self.args.enc_in*3, T=model_pred_len, output_dim=self.args.enc_in*1, hidden_dim=self.args.err_h, device=self.device)
        
        # Automated process to find the best error correction coefficient that was recorded
        possible_coeffs = [0, 0.1, 0.3, 0.5, 0.7, 1] 
        model_loaded = False
        for criterion_name in ["MSE", "MAE"]:
            for coeff in possible_coeffs:
                try:
                    if is_torch_model:
                        modelerr_full.load_state_dict(torch.load(os.path.join(f'{HOME_DIR}/checkpoints/' + setting, f'checkpoint-modelerr-{ecm}-found-best-coeff-{coeff}-season-{self.args.season_coef}-trend-{self.args.trend_coef}-{criterion_name}-ablation1.pth')))
                        modelerr_full.eval()
                        model_loaded = True 
                        break
                except FileNotFoundError:
                    continue

            if not model_loaded:
                raise FileNotFoundError("No valid modelerr checkpoint found for any coeff and criterion.")

            preds = []
            trues = []
            folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_ablation1/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
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
                        if j==0:
                            enc_inp = batch_x
                            oinput = batch_x
                        else:
                            enc_inp =  pred_y
                        batch_x = enc_inp #Autoregression
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        
                        # Forward pass through the model
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
                                perr = pred_full

                            outputs_pred = outputs + perr*self.args.errcor_coef


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
                        
                        if self.args.errcor_coef>0:
                            pred =  outputs_pred[:batch_y.shape[0], -self.args.pred_len:, f_dim:].detach().cpu().numpy()
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
            folder_path = f'{HOME_DIR}/infer_results_seasonal_{self.args.season_coef}_trend_{self.args.trend_coef}_ablation1/' + setting + f'-' + ecm + f'-' + criterion_name + '/'
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
        return

