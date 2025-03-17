from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visualm
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from tqdm import tqdm
import math
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
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

                loss = criterion(pred, true)

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
        criterion = self._select_criterion()

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
            self.model.load_state_dict(torch.load(os.path.join('./scratch/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './scratch/test_results/' + setting + '/'
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
        folder_path = './scratch/results/' + setting + '/'
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
        f = open("./scratch/results/result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    
    def test_infer(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(f'./scratch/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = f'./scratch/infer_results/' + setting + '/'
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
                    # print(input.shape)
                    # print(pred.shape)
                    # print(oinput.shape)
                    # print(true.shape)
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
        folder_path = f'./scratch/infer_results/' + setting + '/'
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
        f = open("./scratch/results/result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def train_infer_batch(self, setting, test=0):
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
        print(setting)
        # setting = "long_term_forecast_ETTh1_96_96_TimesNet_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
        # setting = "long_term_forecast_ETTh2_96_96_TimesNet_ETTh2_ftM_sl96_ll0_pl96_dm32_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
        # setting = "long_term_forecast_ETTh1_96_96_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"
        test_data, test_loader = self._get_data(flag='val')
        data_pred_len =self.args.pred_len
        num_ar = math.ceil(data_pred_len//model_pred_len)
        self.args.pred_len = model_pred_len
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print("data prediction length: ", data_pred_len)
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'./scratch/checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = f'./scratch/infer_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        poses = []
        correction_inputs = []
        correction_targets = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                obatch_y = batch_y.float().to(self.device)
                # print(batch_x.shape)
                # print(obatch_y.shape)
                batch_x_mark = batch_x_mark.float().to(self.device)
                obatch_y_mark = batch_y_mark.float().to(self.device)

                for j in range(num_ar):
                    # print(j)
                    batch_y = obatch_y[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    batch_y_mark = obatch_y_mark[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    if j==0:
                        enc_inp = batch_x
                        oinput = batch_x
                    else:
                        enc_inp =  pred_y
                       
                            

                        # enc_inp = torch.cat([enc_inptp_gt,enc_inptd], dim=2)
                    batch_x = enc_inp #Autoregression
                    
                    # print(batch_x.shape)
                    # print(enc_inp.shape)
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

                    pred_y = outputs
                    if self.args.use_ar==0:
                        pred_y = obatch_y[:, self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]

                    batch_x_mark = batch_y_mark
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    otrue = batch_y[:, :, f_dim:]
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
                    # print(pred.shape)
                    # print(true.shape)
                    # print("--")
                    correction_inputs.append(torch.cat([torch.tensor(batch_x),torch.tensor(pred_y[:, -self.args.pred_len:, f_dim:]).to(self.device)],dim=-1))
                    # correction_inputs.append(torch.tensor(pred).to(self.device))
                    correction_targets.append(torch.tensor(true-pred).to(self.device))
                    preds.append(pred)
                    trues.append(true)
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
                        # print(input.shape)
                        # print(pred.shape)
                        # print(oinput.shape)
                        # print(true.shape)
                        for ii in range(7):
                            gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                            gts.append(gt)
                            pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                            pds.append(pd)
                        visualm(gts, pds, os.path.join(folder_path, f"ar{self.args.use_ar}-{i}-{j}.pdf"))
                    oinput = vbatch_y
                    if i == 20000:
                        break
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'./scratch/infer_results/' + setting + '/'
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
        f = open("./scratch/results/result_long_term_forecast.txt", 'w')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)


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
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size   # 20% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        modelerr = ErrorCorrector(input_dim=X.shape[-1],
                                    T=X.shape[1], 
                                    output_dim=Y.shape[-1],
                                    hidden_dim=self.args.err_h).to(self.device)
        optimizer = torch.optim.Adam(modelerr.parameters(), lr=1e-2)
        # optimizer = optim.RMSprop(modelerr.parameters(), lr=1e-6, alpha=0.99, eps=1e-8)

        # loss_fn = nn.MSELoss()
        loss_fn = nn.SmoothL1Loss()
        # loss_fn = nn.L1Loss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5, verbose=True)

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
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
                torch.save(modelerr.state_dict(), os.path.join(f'./scratch/checkpoints/' + setting, 'checkpoint-modelerr.pth'))

            print("Training complete.")
        return




    def test_infer_batch(self, setting, test=0):
        model_pred_len = self.args.seq_len
        setting_components = setting.split("_")
        print(setting_components)
        setting_components[5] = str(model_pred_len)
        setting_components[11] = "pl"+str(model_pred_len)
        setting = "_".join(setting_components)
        print(setting)
        # setting = "long_term_forecast_ETTh1_96_96_TimesNet_ETTh1_ftM_sl96_ll48_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
        # setting = "long_term_forecast_ETTh2_96_96_TimesNet_ETTh2_ftM_sl96_ll0_pl96_dm32_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
        # setting = "long_term_forecast_ETTh1_96_96_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0"
        test_data, test_loader = self._get_data(flag='test')
        data_pred_len =self.args.pred_len
        num_ar = math.ceil(data_pred_len//model_pred_len)
        self.args.pred_len = model_pred_len
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'./scratch/checkpoints/' + setting, 'checkpoint.pth')))
        
        print('loading modelerr')
        modelerr = ErrorCorrector(input_dim=self.args.enc_in*2,T=model_pred_len, output_dim=self.args.enc_in, hidden_dim=self.args.err_h).to(self.device)
        modelerr.load_state_dict(torch.load(os.path.join(f'./scratch/checkpoints/' + setting, 'checkpoint-modelerr.pth')))
        modelerr.eval()  # Set to evaluation mode
        print("data prediction length: ", data_pred_len)

        preds = []
        trues = []
        folder_path = f'./scratch/infer_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        poses = []
        correction_inputs = []
        correction_targets = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                obatch_y = batch_y.float().to(self.device)
                # print(batch_x.shape)
                # print(obatch_y.shape)
                batch_x_mark = batch_x_mark.float().to(self.device)
                obatch_y_mark = batch_y_mark.float().to(self.device)

                for j in range(num_ar):
                    batch_y = obatch_y[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    batch_y_mark = obatch_y_mark[:,self.args.label_len+model_pred_len*j:self.args.label_len+model_pred_len*(j+1),:]
                    if j==0:
                        enc_inp = batch_x
                        oinput = batch_x
                    else:
                        enc_inp =  pred_y
                       
                            

                    batch_x = enc_inp #Autoregression
                    
                    # print(batch_x.shape)
                    # print(enc_inp.shape)
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
                    
                    f_dim = -1 if self.args.features == 'MS' else 0

                    
                    if self.args.errcor_coef>0:
                        meinput = torch.cat([batch_x, outputs[:, -self.args.pred_len:, f_dim:]], dim=-1)
                        # meinput = outputs[:, -self.args.pred_len:, f_dim:]
                        perr = modelerr(meinput)
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
                    # print(pred.shape)
                    # print(true.shape)
                    # print("--")
                   
                    preds.append(pred)
                    trues.append(true)
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
                        # print(input.shape)
                        # print(pred.shape)
                        # print(oinput.shape)
                        # print(true.shape)
                        for ii in range(7):
                            gt = np.concatenate((oinput[0, :, ii], true[0, :, ii]), axis=0)
                            gts.append(gt)
                            pd = np.concatenate((input[0, :, ii], pred[0, :, ii]), axis=0)
                            pds.append(pd)
                        visualm(gts, pds, os.path.join(folder_path, f"ar{self.args.use_ar}mr{self.args.errcor_coef}-{i}-{j}.pdf"))
                    oinput = vbatch_y
                    if i == 20000:
                        break
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'./scratch/infer_results/' + setting + '/'
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


        return


class ErrorCorrector(nn.Module):
    def __init__(self, input_dim,T, output_dim, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(T, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T),
            nn.Dropout(dropout)
        )
        self.model2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),

        )
        self.output_dim = output_dim

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        x = x.permute(0, 2, 1)        # (B, D, T)
        x = x.reshape(B * D, T)       # (B*D, T)
        out = self.model(x)          # (B*D, T)
        out = out.view(B, D, T)       # (B, D, T)

        return self.model2(out.permute(0, 2, 1))   # back to (B, T, D)