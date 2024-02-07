import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import os
import random
from tqdm import tqdm

from sklearn.metrics import roc_auc_score


def train_c_transformer(model, train_dataset, valid_dataset, exg, epoch, use_treatment=True, batch_size=256, lr=0.0001, weight_decoder=1.0, weight_decay=0.001, tune_lr_every=None, gamma=None, verbose=2,
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_loss = nn.MSELoss()
    c_loss = nn.CrossEntropyLoss()#None if exg.weight is None else torch.Tensor(exg.weight).to(device))
    if tune_lr_every is not None and gamma is not None:
        scheduler = StepLR(optimizer, step_size=tune_lr_every, gamma=gamma)
    
    y_gt_train_all, y_gt_valid_all, y_pred_train_all, y_pred_valid_all, loss_all = [], [], [], [], []
    mae_train_all, mape_train_all, mae_valid_all, mape_valid_all = [], [], [], []
    y_gt_valid_1_all, y_gt_valid_0_all, y_pred_valid_1_all, y_pred_valid_0_all = [], [], [], []
    ate_pred_all, ate_gt_all, t_stat_all = [], [], []
    
    t0 = time.time()
    for epo in range(epoch):
        y_gt_train, y_gt_valid, y_pred_train, y_pred_valid, n_train, loss = 0.0, 0.0, 0.0, 0.0, 0, 0.0
        tt_mae_train,  tt_mape_train, tt_mae_valid, tt_mape_valid = 0.0, 0.0, 0.0, 0.0
        y_pred_valid_1, y_pred_valid_0, y_gt_valid_1, y_gt_valid_0 = 0.0, 0.0, 0.0, 0.0
        n_valid, n_valid_0, n_valid_1 = 0, 0, 0
        
        model.train()
        for x, s, w, y_c, y in train_dl:
            optimizer.zero_grad()
            x, s, w, y_c, y = x.to(device), s.to(device), w.to(device), y_c.to(device), y.to(device)
            if use_treatment:
                x = torch.cat([x, s, w], dim=2)
            else:
                x = torch.cat([x, s], dim=2)
            y_out, y_lt = model(x, s) # prior_mu: batch_size, t0, dim_embedding
            l = weight_decoder * f_loss(y_out, s) + c_loss(y_lt, y_c)
            l.backward()
            optimizer.step()
            
            y_pred = torch.from_numpy(exg.class_to_value(torch.argmax(y_lt, dim=1).to('cpu').detach().numpy())).to(y.device)
            
            y_gt_train += y.sum().to('cpu').item()
            y_pred_train += y_pred.sum()
            
            tt_mae_train += torch.abs(y - y_pred).sum().to('cpu').item()
            tt_mape_train += torch.abs((y - y_pred) / y).sum().to('cpu').item()
            
            n_train += x.shape[0]
            loss += l.to('cpu').item() * x.shape[0]
            
        # validation
        model.eval()
        with torch.no_grad():
            for x, s, w, y_c, y in valid_dl:
                x, s, w, y_c, y = x.to(device), s.to(device), w.to(device), y_c.to(device), y.to(device)
                # print(x.shape, s.shape, w.shape, y_c.shape, y.shape)
                if use_treatment:
                    x = torch.cat([x, s, w], dim=2)
                else:
                    x = torch.cat([x, s], dim=2)
                _, y_lt = model(x, s)
                y_pred = torch.from_numpy(exg.class_to_value(torch.argmax(y_lt, dim=1).to('cpu').detach().numpy())).to(y.device)
                
                y_gt_valid += y.sum().to('cpu').item()
                y_pred_valid += y_pred.sum()
                
                tt_mae_valid += torch.abs(y - y_pred).sum().to('cpu').item()
                tt_mape_valid += torch.abs((y - y_pred) / y).sum().to('cpu').item()
                
                y_gt_valid_1 += y[torch.where(w[:, 0, 0]==1)].sum().to('cpu').item()
                y_gt_valid_0 += y[torch.where(w[:, 0, 0]==0)].sum().to('cpu').item()    
                
                y_pred_valid_1 += y_pred[torch.where(w[:, 0, 0]==1)].sum().to('cpu').item()
                y_pred_valid_0 += y_pred[torch.where(w[:, 0, 0]==0)].sum().to('cpu').item()                
                
                n_valid += x.shape[0]
                n_valid_1 += (w[:, 0, 0]==1).sum().to('cpu').item()
                n_valid_0 += (w[:, 0, 0]==0).sum().to('cpu').item()
        
        if tune_lr_every is not None and gamma is not None:
            scheduler.step()

        mae_train_all.append(tt_mae_train / n_train)
        mape_train_all.append(tt_mape_train / n_train)
        mae_valid_all.append(tt_mae_valid / n_valid)
        mape_valid_all.append(tt_mape_valid / n_valid)
        
        y_gt_train_all.append(y_gt_train / n_train)
        y_gt_valid_all.append(y_gt_valid / n_valid)
        y_pred_train_all.append(y_pred_train / n_train)
        y_pred_valid_all.append(y_pred_valid / n_valid)
        loss_all.append(loss / n_train)
        
        y_gt_valid_1_all.append(y_gt_valid_1 / n_valid_1)
        y_gt_valid_0_all.append(y_gt_valid_0 / n_valid_0)
        y_pred_valid_1_all.append(y_pred_valid_1 / n_valid_1)
        y_pred_valid_0_all.append(y_pred_valid_0 / n_valid_0)
        
        ate_gt_all.append(y_gt_valid_1 / n_valid_1 - y_gt_valid_0 / n_valid_0)
        ate_pred_all.append(y_pred_valid_1 / n_valid_1 - y_pred_valid_0 / n_valid_0)
        
        if (epo + 1) % verbose == 0:
            print("[%.1f sec], epoch: %d, loss: %.3f, mae train: %.3f, mape train: %.1f%%, y mae train: %.3f, y mape train: %.2f%%, mae valid: %.3f, mape valid: %.1f%%, y mae valid: %.3f, y mape valid: %.2f%%, mae ate: %.4f, mape ate: %.3f%%" % 
                  (time.time() - t0, epo + 1, loss_all[-1], mae_train_all[-1], mape_train_all[-1] * 100, 
                   (y_pred_train - y_gt_train) / n_train, abs(y_gt_train - y_pred_train)/ y_gt_train * 100,
                   mae_valid_all[-1], mape_valid_all[-1] * 100, 
                   (y_pred_valid - y_gt_valid) / n_valid, abs(y_gt_valid - y_pred_valid) / y_gt_valid * 100,
                   ate_pred_all[-1] - ate_gt_all[-1], abs((ate_gt_all[-1] - ate_pred_all[-1]) / ate_gt_all[-1] * 100)))
            
    return mae_train_all, mape_train_all, y_gt_train_all, y_pred_train_all, mae_valid_all, mape_valid_all, y_gt_valid_all, y_pred_valid_all, ate_gt_all, ate_pred_all, loss_all




def train_r_transformer(model, train_dataset, valid_dataset, epoch, batch_size=512, lr=0.0001, decoder_weight=1.0, weight_decay=0.001, shifted=False, tune_lr_every=None, gamma=None, verbose=2):
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_loss = nn.MSELoss()
    c_loss = nn.BCEWithLogitsLoss()

    if tune_lr_every is not None and gamma is not None:
        scheduler = StepLR(optimizer, step_size=tune_lr_every, gamma=gamma)
    
    y_gt_train_all, y_gt_valid_all, y_pred_train_all, y_pred_valid_all, loss_all = [], [], [], [], []
    mae_train_all, mape_train_all, mae_valid_all, mape_valid_all = [], [], [], []
    y_gt_valid_1_all, y_gt_valid_0_all, y_pred_valid_1_all, y_pred_valid_0_all = [], [], [], []
    ate_pred_all, ate_gt_all, t_stat_all = [], [], []
    
    t0 = time.time()
    
    for epo in range(epoch):
        y_gt_train, y_gt_valid, y_pred_train, y_pred_valid, n_train, loss = 0.0, 0.0, 0.0, 0.0, 0, 0.0
        tt_mae_train,  tt_mape_train, tt_mae_valid, tt_mape_valid = 0.0, 0.0, 0.0, 0.0
        y_pred_valid_1, y_pred_valid_0, y_gt_valid_1, y_gt_valid_0 = 0.0, 0.0, 0.0, 0.0
        n_valid, n_valid_0, n_valid_1 = 0, 0, 0
        
        model.train()
        for x, s, w, y in train_dl:
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            x, s, w, y = x.to(device), s.to(device), w.to(device), y.to(device)
            x = torch.cat([x, s, w], dim=2)

            if shifted:
                s_dec = torch.ones([s.shape[0], 1, s.shape[2]]).float().to(s.device)
                s_dec = torch.cat([s_dec, s[:, :-1, :]], dim=1)
                y_out, y_lt = model(x, s_dec) # prior_mu: batch_size, t0, dim_embedding
            else:
                y_out, y_lt = model(x, s)
            
            # print(y_out.shape, s.shape, y_lt.shape, y.shape)
            # break
            
            l =  f_loss(y_lt, y) + decoder_weight * f_loss(y_out, s) 
            l.backward()
            optimizer.step()
            
            y_gt_train += y.sum().to('cpu').item()
            y_pred_train += y_lt.sum().to('cpu').item()
            tt_mae_train += torch.abs(y - y_lt).sum().to('cpu').item()
            tt_mape_train += torch.abs((y - y_lt) / y).sum().to('cpu').item()
            
            n_train += x.shape[0]
            loss += l.to('cpu').item() * x.shape[0]
            
        # validation
        model.eval()
        with torch.no_grad():
            for x, s, w, y in valid_dl:
                x, s, w, y = x.to(device), s.to(device), w.to(device), y.to(device)
                y = y.unsqueeze(1)
                x = torch.cat([x, s, w], dim=2)
                
                s_dec = torch.ones([s.shape[0], 1, s.shape[2]]).float().to(s.device)
                s_dec = torch.cat([s_dec, s[:, :-1, :]], dim=1)
                
                if shifted:
                    s_dec = torch.ones([s.shape[0], 1, s.shape[2]]).float().to(s.device)
                    s_dec = torch.cat([s_dec, s[:, :-1, :]], dim=1)
                    _, y_lt = model(x, s_dec) # prior_mu: batch_size, t0, dim_embedding
                else:
                    _, y_lt = model(x, s)
                
                y_gt_valid += y.sum().to('cpu').item()
                y_pred_valid += y_lt.sum().to('cpu').item()
                tt_mae_valid += torch.abs(y - y_lt).sum().to('cpu').item()
                tt_mape_valid += torch.abs((y - y_lt) / y).sum().to('cpu').item()
                
                y_gt_valid_1 += y[torch.where(w[:, 0, 0]==1)].sum().to('cpu').item()
                y_gt_valid_0 += y[torch.where(w[:, 0, 0]==0)].sum().to('cpu').item()    
                
                y_pred_valid_1 += y_lt[torch.where(w[:, 0, 0]==1)].sum().to('cpu').item()
                y_pred_valid_0 += y_lt[torch.where(w[:, 0, 0]==0)].sum().to('cpu').item()                
                
                n_valid += x.shape[0]
                n_valid_1 += (w[:, 0, 0]==1).sum().to('cpu').item()
                n_valid_0 += (w[:, 0, 0]==0).sum().to('cpu').item()
        
        if tune_lr_every is not None and gamma is not None:
            scheduler.step()
                
        mae_train_all.append(tt_mae_train / n_train)
        mape_train_all.append(tt_mape_train / n_train)
        mae_valid_all.append(tt_mae_valid / n_valid)
        mape_valid_all.append(tt_mape_valid / n_valid)
        
        y_gt_train_all.append(y_gt_train / n_train)
        y_gt_valid_all.append(y_gt_valid / n_valid)
        y_pred_train_all.append(y_pred_train / n_train)
        y_pred_valid_all.append(y_pred_valid / n_valid)
        loss_all.append(loss / n_train)
        
        y_gt_valid_1_all.append(y_gt_valid_1 / n_valid_1)
        y_gt_valid_0_all.append(y_gt_valid_0 / n_valid_0)
        y_pred_valid_1_all.append(y_pred_valid_1 / n_valid_1)
        y_pred_valid_0_all.append(y_pred_valid_0 / n_valid_0)
        
        ate_gt_all.append(y_gt_valid_1 / n_valid_1 - y_gt_valid_0 / n_valid_0)
        ate_pred_all.append(y_pred_valid_1 / n_valid_1 - y_pred_valid_0 / n_valid_0)
        
        if (epo + 1) % verbose == 0:
            print("[%.1f sec], epoch: %d, loss: %.4f, mae train: %.3f, mape train: %.1f%%, y mae train: %.3f, y mape train: %.2f%%, mae valid: %.3f, mape valid: %.1f%%, y mae valid: %.3f, y mape valid: %.2f%%, mae ate: %.4f, mape ate: %.3f%%" % 
                  (time.time() - t0, epo + 1, loss_all[-1], mae_train_all[-1], mape_train_all[-1] * 100, 
                   (y_pred_train - y_gt_train) / n_train, abs(y_gt_train - y_pred_train)/ y_gt_train * 100,
                   mae_valid_all[-1], mape_valid_all[-1] * 100, 
                   (y_pred_valid - y_gt_valid) / n_valid, abs(y_gt_valid - y_pred_valid) / y_gt_valid * 100,
                   ate_pred_all[-1] - ate_gt_all[-1], abs((ate_gt_all[-1] - ate_pred_all[-1])) / ate_gt_all[-1] * 100))
            
    return mae_train_all, mape_train_all, y_gt_train_all, y_pred_train_all, mae_valid_all, mape_valid_all, y_gt_valid_all, y_pred_valid_all, ate_gt_all, ate_pred_all, loss_all