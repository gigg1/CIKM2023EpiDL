# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
# from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
from models import *
from data import *

import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter

import torch.nn as nn
import os
import matplotlib.pyplot as plt

from PlotFunc import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='japan', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='japan-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=32, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.6, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.2, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='cola_gnn', choices=['cola_gnn_epi', 'cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn'], help='')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=False, help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=False,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=1, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=3,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=200, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')
ap.add_argument('--epilambda', type=float, default=0.2, help='the weights of epidemiological loss')

args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dcrnn_model import *

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
if args.model == 'cola_gnn_epi':
    log_token = '%s.%s.w-%s.h-%s.%s.lam-%s.epo-%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model, args.epilambda, args.epochs)
else:
    log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

data_loader = DataBasicLoader(args)

# print (data_loader.rawdat)
# print (data_loader.max.shape)
# print (data_loader.max)
# print (data_loader.min)
# print (data_loader.max-data_loader.min)
# print (data_loader.val)

if args.model == 'CNNRNN_Res':
    model = CNNRNN_Res(args, data_loader)  
elif args.model == 'RNN':
    model = RNN(args, data_loader)
elif args.model == 'AR':
    model = AR(args, data_loader)
elif args.model == 'ARMA':
    model = ARMA(args, data_loader)
elif args.model == 'VAR':
    model = VAR(args, data_loader)
elif args.model == 'GAR':
    model = GAR(args, data_loader)
elif args.model == 'SelfAttnRNN':
    model = SelfAttnRNN(args, data_loader)
elif args.model == 'lstnet':
    model = LSTNet(args, data_loader)      
elif args.model == 'stgcn':
    model = STGCN(args, data_loader, data_loader.m, 1, args.window, 1)  
elif args.model == 'dcrnn':
    model = DCRNNModel(args, data_loader)   
elif args.model == 'cola_gnn':
    model = cola_gnn(args, data_loader)     
elif args.model == 'cola_gnn_epi':
    model = cola_gnn_epi(args, data_loader)
else: 
    raise LookupError('can not find the model')

logger.info('model %s', model)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)

def evaluate(data_loader, data, modelName, tag='val'):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []

    # print ("1111111")
    # print (data)

    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        
        if modelName == "cola_gnn_epi":
            output, EpiOutput, _, _, _  = model(X)
        else:
            output,_  = model(X)

        loss_train = F.l1_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m);

        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx) # [n_samples, 47]

    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  #(#n_samples, 47)

    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47

    # mseList1=[]
    # tmp = 0
    # for loc in range(0, y_true_states.shape[1]):
    #     # print ("loc:", loc)
    #     # print (y_true_states[:,loc].shape, y_pred_states[:,loc].shape)
    #     tmp = tmp + mean_squared_error(y_true_states[:,loc], y_pred_states[:,loc])
    #     mseList1.append(mean_squared_error(y_true_states[:,loc], y_pred_states[:,loc]))

    # tmp1 = np.sqrt(tmp/y_true_states.shape[1])
    # mselist2 = mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')

    # print ("")
    # print (mseList1)
    # print (mselist2)
    # print (np.sqrt(mselist2))
    # print ("calculated 1:", tmp1)
    # print ("rmse_states", rmse_states)

    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
    
    pcc_tmp = []

    for k in range(data_loader.m):
        pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])

    pcc_states = np.mean(np.array(pcc_tmp))

    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))

    # print (y_true.shape)
    # print (y_pred.shape)

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc = pearsonr(y_true,y_pred)[0]
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    # print ("rmse", rmse)

    global y_true_t
    global y_pred_t

    y_true_t = y_true_states
    y_pred_t = y_pred_states

    return float(total_loss / n_samples), mae, std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae

def train(data_loader, data, modelName, Lambda):
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        locationNumber = X.shape[2]
        numberOfSamples = X.shape[0]

        optimizer.zero_grad()
        if modelName == "cola_gnn_epi":
            output, EpiOutPut, _, _, _ = model(X)
        else:
            output, _ = model(X)

        # print ("")
        # print (X)
        # print (output)

        if Y.size(0) == 1:
            Y = Y.view(-1)
            # EpiOutPut = EpiOutPut.view(-1)
        
        # --------------------------------------------
        # modified loss with epidemiological constrains
        if modelName == "cola_gnn_epi":
            loss_train = F.l1_loss(output, Y) + Lambda*criterion(EpiOutPut,Y)
        else:
            loss_train = F.l1_loss(output, Y) # mse_loss

        total_loss += loss_train.item()
        # --------------------------------------------

        torch.autograd.set_detect_anomaly(True)
        loss_train.backward()
        # loss_train.backward(retain_graph=True)
        optimizer.step()

        # --------------------------------------------
        # ------------ set the parameter constraints
        # print ("**********************")
        # print (model.parameters())

        # if args.model == 'cola_gnn_epi':
        #     for name, para in model.named_parameters():
        #         # print (name, ":",para.size())
        #         if name == "Beta" or name == "Gamma":
        #             para.data.clamp_(min=0, max=1)
        #         if name == "mask_mat":
        #             para.data.clamp_(min=0, max=None)
        #             para.data.clamp_(min=0, max=1)

                # print (self.Beta)
                # self.Beta=Parameter(torch.clamp(self.Beta, min=0, max=1))
                # print (self.Beta)
        # print ("**********************")
        # --------------------------------------------

        n_samples += (output.size(0) * data_loader.m)
    return float(total_loss / n_samples)
 
bad_counter = 0
best_epoch = 0
best_val = 1e+20;

criterion = nn.MSELoss(reduction='sum');

ifPlot = 0

try:
    print('begin training');

    # --------------------------------------------
    # Plot the convergence
    x_epoch=[]
    y_train_loss=[]
    y_validate_loss=[]
    # --------------------------------------------

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train, args.model, args.epilambda)
        val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val, args.model)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))

        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss}, epoch )
            writer.add_scalars('data/loss', {'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)

        # --------------------------------------------
        # Plot the convergence
        x_epoch.append(epoch)
        y_train_loss.append(train_loss)
        y_validate_loss.append(val_loss)
        # --------------------------------------------
       
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = '%s/%s.pt' % (args.save_dir, log_token)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Best validation epoch:',epoch, time.ctime());
            test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test, args.model, tag='test')
            # print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
            print('TEST MAE {:5.4f} | std {:5.4f} | RMSE {:5.4f} | RMSEs {:5.4f} | PCC {:5.4f} | PCCs {:5.4f} | R2 {:5.4f} | R2s {:5.4f} | Var {:5.4f} | Vars {:5.4f} | Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # --------------------------------------------
    # Plot the convergence
    if ifPlot==1:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax2 = ax.twinx()
        labelSize=12
        ax.plot(x_epoch,y_train_loss,'o-',alpha=0.7,label="Train loss",color="brown",markersize=3,linewidth=2)
        ax2.plot(x_epoch,y_validate_loss,'o-',alpha=0.7,label="Validation loss",color="navy",markersize=3,linewidth=2)
        ax.legend(loc=2)
        ax2.legend(loc=1)
        ax.set_xlabel("Epoch",fontsize=labelSize)
        ax.set_ylabel("Train Loss",fontsize=labelSize)
        ax2.set_ylabel("Validation Loss",fontsize=labelSize)
        # ax2.set_ylabel("Loss",fontsize=labelSize)

        # figure_save_dir="./Results/"
        save_name=args.model+"_"+"loss"
        figure_save_dir = './Figures/%s/' % (log_token)
        
        if not os.path.exists(figure_save_dir):
            os.makedirs(figure_save_dir)

        plt.savefig(figure_save_dir+save_name+".pdf",bbox_inches='tight',transparent=True)
        plt.close('all')
    # --------------------------------------------

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch',epoch)

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test, args.model, tag='test')
print('Final evaluation')
print('TEST MAE {:5.4f} | std {:5.4f} | RMSE {:5.4f} | RMSEs {:5.4f} | PCC {:5.4f} | PCCs {:5.4f} | R2 {:5.4f} | R2s {:5.4f} | Var {:5.4f} | Vars {:5.4f} | Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))

####################################################
# ---- get the prediction
####################################################
def GetPrediction(model, data_loader, data, modelName, tag='Prediction'):
    model.eval()

    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    X_true_mx = []

    if modelName == 'cola_gnn_epi':
        BetaList = []
        GammaList = []
        NGMList = []

    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        if modelName == 'cola_gnn_epi':
            output, EpiOutput, Beta, Gamma, NGMT  = model(X)
        else:
            output, _  = model(X)

        X_true_mx.append(X)
        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

        if modelName == 'cola_gnn_epi':
            BetaList.append(Beta.cpu().detach())
            GammaList.append(Gamma.cpu().detach())
            NGMList.append(NGMT.cpu().detach())

    X_true_mx = torch.cat(X_true_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx) # [n_samples, 47]

    if modelName == 'cola_gnn_epi':
        BetaList = torch.cat(BetaList).numpy()
        GammaList = torch.cat(GammaList).numpy()
        NGMList = torch.cat(NGMList).numpy()

    # Restore data to real values
    X_true_states = X_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  #(#n_samples, 47)

    # Time * location
    # print ("--------- X_true")
    # print (X_true_states.shape)
    # print ("--------- y_true")
    # print (y_true_states.shape)
    # print ("--------- y_pred")
    # print (y_pred_states.shape)
    # print ("--------- NextGenerationMatrix")
    # print (NextGenerationMatrix.shape)
    if modelName == 'cola_gnn_epi':
        return X_true_states, y_true_states, y_pred_states, BetaList, GammaList, NGMList
    else:
        return X_true_states, y_true_states, y_pred_states

if ifPlot == 1 and args.model == 'cola_gnn_epi':
    # plot trends of predicted case and real case
    X_true, y_true, y_pred, BetaList, GammaList, NGMList = GetPrediction(model, data_loader, data_loader.test, args.model, tag='Prediction')
    save_dir = './Figures/%s/' % (log_token)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    PlotPredictionTrends(y_true.T, y_pred.T, save_dir)
    PlotParameters(BetaList.T, GammaList.T, save_dir)
    PlotTrends(X_true.transpose(2, 0, 1), y_true.T, y_pred.T, save_dir, args.horizon)

    Type = "Next Generation Matrix"
    SaveName = "NGM"
    # PlotEachMatrix(NGMList, Type, SaveName, save_dir)
    PlotAllMatrices(NGMList, Type, SaveName, save_dir)

