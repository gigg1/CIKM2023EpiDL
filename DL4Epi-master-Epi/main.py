#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time

import torch
import torch.nn as nn
# from models import AR, VAR, GAR, RNN, VAR_mask
# from models import CNNRNN, CNNRNN_Res, CNNRNN_Res_epi
from models import CNNRNN_Res_epi
import numpy as np
import sys
import os
from utils import *
from utils_ModelTrainEval import *
import Optim

import matplotlib.pyplot as plt

from PlotFunc import *

import xlwt
import csv

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Epidemiology Forecasting')
# --- Data option
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--train', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--valid', type=float, default=0.2,help='how much data used for validation')
parser.add_argument('--model', type=str, default='CNNRNN_Res_epi',help='model to select')
# --- CNNRNN option
parser.add_argument('--sim_mat', type=str,help='file of similarity measurement (Required for CNNRNN_Res_epi)')
parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units')
parser.add_argument('--residual_window', type=int, default=4,help='The window size of the residual component')
parser.add_argument('--ratio', type=float, default=1.,help='The ratio between CNNRNN and residual')
parser.add_argument('--output_fun', type=str, default=None, help='the output function of neural net')
# --- Logging option
parser.add_argument('--save_dir', type=str,  default='./save',help='dir path to save the final model')
parser.add_argument('--save_name', type=str,  default='tmp', help='filename to save the final model')
# --- Optimization option
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--epochs', type=int, default=100,help='upper epoch limit')
parser.add_argument('--clip', type=float, default=1.,help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 regularization)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='batch size')
# --- Misc prediction option
parser.add_argument('--horizon', type=int, default=12, help='predict horizon')
parser.add_argument('--window', type=int, default=24 * 7,help='window size')
parser.add_argument('--metric', type=int, default=1, help='whether (1) or not (0) normalize rse and rae with global variance/deviation ')
parser.add_argument('--normalize', type=int, default=0, help='the normalized method used, detail in the utils.py')

parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=None, help='GPU number to use')
parser.add_argument('--cuda', type=str, default=None, help='use gpu or not')

parser.add_argument('--epilambda', type=float, default=0.2, help='the weights of epidemiological loss')

args = parser.parse_args()
print(args);
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.model in ['CNNRNN_Res_epi'] and args.sim_mat is None:
    print('CNNRNN_Res_epi requires "sim_mat" option')
    sys.exit(0)

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args);
# print (Data)

model = eval(args.model).Model(args, Data);
print('model:', model)
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

criterion = nn.MSELoss(reduction='sum');
evaluateL2 = nn.MSELoss(reduction='sum');
evaluateL1 = nn.L1Loss(reduction='sum')

if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();

best_val = 10000000;
# for name,para in model.named_parameters():
#     print (name,para.size())
#     print (para)
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, model.named_parameters(), weight_decay = args.weight_decay,
)

ifPlot = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training');
    
    # --------------------------------------------
    # Plot the convergence
    x_epoch=[]
    y_train_loss=[]
    y_validate_loss=[]
    # --------------------------------------------

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size, args.model, args.epilambda)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size, args.model);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        
        if math.isnan(train_loss)==True:
            sys.exit()
        
        # --------------------------------------------
        # Plot the convergence
        x_epoch.append(epoch)
        y_train_loss.append(train_loss)
        y_validate_loss.append(val_loss)
        # --------------------------------------------

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            model_path = '%s/%s.pt' % (args.save_dir, args.save_name)

            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('best validation');
            test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
        
        #     y_test_loss.append(y_test_loss[-1])
        # else:
        #     y_test_loss.append(y_test_loss[-1])

    # # --------------------------------------------
    # # Plot the convergence
    if ifPlot == 1:
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

        # figure_save_dir="./Results/"
        save_name=args.model+"_"+"loss"
        model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
        figure_save_dir = './Figures/%s/' % (args.save_name)
        
        if not os.path.exists(figure_save_dir):
            os.makedirs(figure_save_dir)

        plt.savefig(figure_save_dir+save_name+".pdf",bbox_inches='tight',transparent=True)
        plt.close('all')
    # # --------------------------------------------

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# print (args.save_dir)
# print (args.save_name)
model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));

print ("----------------------------")
print ("Data.test")
test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model);
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

# # --------------------------------------------
if args.model=="CNNRNN_Res_epi" and ifPlot == 1:
    X_true, Y_predict, Y_true, BetaList, GammaList, NGMList = GetPrediction(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model)
    # print (X_true.shape)
    # print (Y_predict.shape)
    # print (Y_true.shape)

    save_dir = './Figures/%s.epo-%s/' % (args.save_name, args.epochs)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    PlotPredictionTrends(Y_true.T, Y_predict.T, save_dir)
    PlotParameters(BetaList.T, GammaList.T, save_dir)
    PlotTrends(X_true.transpose(2, 0, 1), Y_true.T, Y_predict.T, save_dir, args.horizon)
    
    Type = "Next Generation Matrix"
    SaveName = "NGM"
    # PlotEachMatrix(NGMList, Type, SaveName, save_dir)
    PlotAllMatrices(NGMList, Type, SaveName, save_dir)


