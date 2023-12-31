#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time

import torch
import torch.nn as nn
from models import AR, VAR, GAR, RNN, VAR_mask
from models import CNNRNN, CNNRNN_Res, CNNRNN_Res_epi
import numpy as np
import sys
import os
from utils import *
import Optim

import matplotlib.pyplot as plt

import xlwt
import csv

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        output = model(X);

        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict,output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)

        # print ("X")
        # print (X)
        # print (X.shape)
        # print ("Y")
        # print (Y * scale)
        # print (Y.shape)
        # print ("Output")
        # print (output * scale)
        # print (output.shape)
        
        if torch.__version__ < '0.4.0':
            total_loss += evaluateL2(output * scale , Y * scale ).data[0]
            total_loss_l1 += evaluateL1(output * scale , Y * scale ).data[0]
        else:
            total_loss += evaluateL2(output * scale , Y * scale ).item()
            total_loss_l1 += evaluateL1(output * scale , Y * scale ).item()
        
        n_samples += (output.size(0) * loader.m);

        rmselist=[]
        outputValue = (output * scale).T
        RealValue = (Y * scale).T
        maelist=[]
        for i in range(0,len(outputValue)):
            LengthOfTime=len(outputValue[i])
            # print (outputValue[i])
            # print (RealValue[i])
            
            # print (LengthOfTime)
            # print (math.sqrt(evaluateL2(outputValue[i] , RealValue[i] ).item()/LengthOfTime))
            rmselist.append(math.sqrt(evaluateL2(outputValue[i] , RealValue[i] ).item()/LengthOfTime))

    # print (rmselist)
    # print (total_loss)
    rse = math.sqrt(total_loss / n_samples)/loader.rse
    rae = (total_loss_l1/n_samples)/loader.rae
    correlation = 0;

    predict = predict.data.numpy();
    Ytest = test.data.numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    # root-mean-square error, absolute error, correlation
    return rse, rae, correlation;

def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        output = model(X);

        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict,output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)

    # predict = predict.data.numpy();
    # Ytest = test.data.numpy();

    outputValue = (output * scale).T
    RealValue = (Y * scale).T

    return outputValue, RealValue

def train(loader, data, model, criterion, optim, batch_size, modelName, Lambda):
    model.train();
    total_loss = 0;
    n_samples = 0;
    counter = 0
    for inputs in loader.get_batches(data, batch_size, True):
        counter += 1
        X, Y = inputs[0], inputs[1]
        
#        print ("-----------")
#        print (X)
#        print (X.shape)
#        print (Y)
#        print (Y.shape)

        model.zero_grad();
        output = model(X);
        scale = loader.scale.expand(output.size(0), loader.m)
        # loss = criterion(output * scale, Y * scale);
        
        # --------------------------------------------
        # modified loss with epidemiological constrains
        # print ("----------- Check point 2 -----------")
        # print (model.NextGenerationMatrix.shape)
        # print (X[:,-1,:].shape)
        # print (Y.shape)

        if modelName == "CNNRNN_Res_epi":
            # Lambda=0.2
            # print (model.NextGenerationMatrix.dtype)
            loss = criterion(output * scale, Y * scale)+Lambda*criterion(X[:,-1,:].matmul(model.NextGenerationMatrix.T), Y);
        else:
            loss = criterion(output * scale, Y * scale);
        # --------------------------------------------

        loss.backward(retain_graph=True);
        optim.step();

        # --------------------------------------------
        # ------------ set the parameter constraints
        # print ("**********************")
        # print (model.parameters())
        if modelName == "CNNRNN_Res_epi":
            for name, para in model.named_parameters():
                # print (name, ":",para.size())
                if name == "Beta" or name == "Gamma" or name == "Mu":
                    para.data.clamp_(min=0, max=1)
                if name == "mask_mat":
                    para.data.clamp_(min=0, max=None)

                # print (self.Beta)
                # self.Beta=Parameter(torch.clamp(self.Beta, min=0, max=1))
                # print (self.Beta)
        # print ("**********************")
        # --------------------------------------------

        if torch.__version__ < '0.4.0':
            total_loss += loss.data[0]
        else:
            total_loss += loss.item()
        n_samples += (output.size(0) * loader.m);
    return total_loss / n_samples

parser = argparse.ArgumentParser(description='Epidemiology Forecasting')
# --- Data option
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--train', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--valid', type=float, default=0.2,help='how much data used for validation')
parser.add_argument('--model', type=str, default='AR',help='model to select')
# --- CNNRNN option
parser.add_argument('--sim_mat', type=str,help='file of similarity measurement (Required for CNNRNN, CNN)')
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
if args.model in ['CNNRNN', 'CNN', 'VAR_mask'] and args.sim_mat is None:
    print('CNNRNN/CNN/VAR_mask requires "sim_mat" option')
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
print (Data)

model = eval(args.model).Model(args, Data);
print('model:', model)
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();

best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
)

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
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size,args.model,args.epilambda)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size);
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
            test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
        
        #     y_test_loss.append(y_test_loss[-1])
        # else:
        #     y_test_loss.append(y_test_loss[-1])

    # --------------------------------------------
    # Plot the convergence
    fig, ax = plt.subplots(figsize=(15, 8))
    labelSize=12
    ax.plot(x_epoch,y_train_loss,'o-',alpha=0.7,label="Train loss",color="brown",markersize=3,linewidth=2)
    ax.plot(x_epoch,y_validate_loss,'o-',alpha=0.7,label="Validation loss",color="navy",markersize=3,linewidth=2)
    ax.legend()
    ax.set_xlabel("Epoch",fontsize=labelSize)
    ax.set_ylabel("Loss",fontsize=labelSize)

    figure_save_dir="./results/"
    save_name=args.model+"_"+"loss"
    plt.savefig(figure_save_dir+save_name+".pdf",bbox_inches='tight',transparent=True)
    plt.close('all')
    # --------------------------------------------

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
test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

# ifprint=1
ifprint=0
# --------------------------------------------
# predict, Ytest=GetPrediction(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
# print the leaned parameters
if ifprint==1:
    print ("----------- Mask matrix")
    print (model.mask_mat)
    print ("----------- Adjacent matrix")
    print (model.adj)
# print (predict)
# print (predict.shape)
# print (Ytest)
# print (Ytest.shape)

# YP=predict.detach().numpy()
# YR=Ytest.detach().numpy()
# print (YP.shape())
# print (YR.shape())

if args.model=="CNNRNN_Res_epi":
    if ifprint==1:
        # Beta_vector = torch.zeros([model.m, 1],dtype=torch.float64)
        # Gamma_vector = torch.zeros([model.m, 1],dtype=torch.float64)
        # Mu_vector = torch.zeros([model.m, 1],dtype=torch.float64)

        # for i in range(0,model.m):
        #     Beta_vector[i] = model.Beta[i][i]
        #     Gamma_vector[i] = model.Gamma[i][i]
        #     Mu_vector[i] = model.Mu[i][i]

        print ("----------- Check point 3 -----------")
        print ("----------- Beta")
        print (model.Beta)
        # print (Beta_vector)
        print ("----------- Gamma")
        print (model.Gamma)
        # print (Gamma_vector)
        print ("----------- Mu")
        print (model.Mu)
        # print (Mu_vector)
        print ("----------- Next generation matrix")
        print (model.NextGenerationMatrix)

# --------------------------------------------


