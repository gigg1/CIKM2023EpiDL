import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

import torch
import torch.nn as nn

from utils_ModelTrainEval import *
from utils import *

from models import AR, VAR, GAR, RNN, VAR_mask
from models import CNNRNN, CNNRNN_Res, CNNRNN_Res_epi

import argparse

def PlotTrends(RealLocationTimeData, PredictedLocationTimeData, save_dir):
    print (RealLocationTimeData)
    print (RealLocationTimeData.shape)

    LocationNumber = RealLocationTimeData.shape[0]
    TimeLength = RealLocationTimeData.shape[1]

    for i in range(0,LocationNumber):
        x = [count for count in range(0,TimeLength)]
        y = RealLocationTimeData[i]
        y2 = PredictedLocationTimeData[i]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x,y,color='tab:blue',label="GroundTruth")
        ax.plot(x,y2,color='tab:orange',label="Prediction")

        # set the limits
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
        plt.legend()
        ax.set_title("Location " + str(i+1))
        ax.set_xlabel("Week")
        ax.set_ylabel("Influenza activity level")
        FigureType="PredictionAndGroundtruth"
        plt.savefig(save_dir+FigureType+"-"+str(i+1)+".pdf",transparent=True,bbox_inches='tight')
        # plt.show()
        plt.close()

def PlotMatrix(Matrix, Type, SaveName, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if Type == "Next Generation Matrix" or "Human Mobility Matrix":
    	cmap='Spectral_r'
    else:
    	# cmap='Spectral_r'
	    cmap='RdBu_r'
	    # cmap='viridis'
	    # cmap='summer'

	# vmin=minValue, vmax=maxValue

    ax = sns.heatmap(data=Matrix,fmt=".2f",cmap=cmap,annot=False,
    					square=True, cbar=False, xticklabels=False, yticklabels=False)

    ax = sns.heatmap(data=Matrix,fmt=".2f",cmap=cmap,annot=False,
                        square=True, cbar=True, xticklabels=False, yticklabels=False)

    # xticklabels=2, yticklabels=2
    # ax = sns.heatmap(HeatMapList[i],fmt=".2f",cmap='RdBu_r',annot=False,vmin=minValue,vmax=maxValue)
    ax.xaxis.tick_top()
    # ax.set_title(Type)
    
    # plt.xticks(fontsize=3,rotation = 60)
    # plt.yticks(fontsize=3,rotation = 60)
    plt.savefig(save_dir+SaveName+".pdf",transparent=True,bbox_inches='tight')
    plt.close()

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

Data = Data_utility(args);
# print (Data)

model = eval(args.model).Model(args, Data);

# Load the model
model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
# ModelFile = "./save/cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt"+".pt"
save_dir = './Figures/%s/' % (args.save_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(model_path, 'rb') as f:
	model.load_state_dict(torch.load(f));

# evaluateL2 = nn.MSELoss(size_average=False);
# evaluateL1 = nn.L1Loss(size_average=False)

evaluateL2 = nn.MSELoss(reduction='sum');
evaluateL1 = nn.L1Loss(reduction='sum')

print ("----------------------------")
print ("Data.test")
test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

predict, Ytest=GetPrediction(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size)
# print ("-----------")
# print (predict)
# print (predict.shape)
# print ("-----------")
# print (Ytest)
# print (Ytest.shape)

# YP=predict.detach().numpy()
# YR=Ytest.detach().numpy()
# print (YP.shape())
# print (YR.shape())

PlotTrends(Ytest.detach().numpy(), predict.detach().numpy(),save_dir)

ifprint=1
# ifprint=0

if args.model=="CNNRNN_Res_epi":
    if ifprint==1:
        # Beta_vector = torch.zeros([model.m, 1],dtype=torch.float64)
        # Gamma_vector = torch.zeros([model.m, 1],dtype=torch.float64)
        # Mu_vector = torch.zeros([model.m, 1],dtype=torch.float64)

        # for i in range(0,model.m):
        #     Beta_vector[i] = model.Beta[i][i]
        #     Gamma_vector[i] = model.Gamma[i][i]
        #     Mu_vector[i] = model.Mu[i][i]
        
        # print the leaned parameters
        print ("----------- Mask matrix")
        print (model.mask_mat)
        print ("----------- Adjacent matrix")
        print (model.adj)
        
        print ("----------- Check point 3 -----------")
        print ("----------- Beta")
        print (model.Beta)
        # print (Beta_vector)
        print ("----------- Gamma")
        print (model.Gamma)
        # print (Gamma_vector)
        # print ("----------- Mu")
        # print (model.Mu)
        # print (Mu_vector)
        print ("----------- Next generation matrix")
        print (model.NextGenerationMatrix)

    Type = "Next Generation Matrix"
    SaveName = "NextGenerationMatrix"
    PlotMatrix(model.NextGenerationMatrix.detach().numpy(), Type, SaveName, save_dir)

    D = model.mask_mat * model.adj
    D[D < 0] = 0
    D[D > 1] = 1
    Type = "Human Mobility Matrix"
    SaveName = "HumanMobilityMatrix"
    PlotMatrix(D.detach().numpy(), Type, SaveName, save_dir)

    # Type = "Random Vector"
    # SaveName = "RandomVector1"
    # vector1 = np.random.randn(29).reshape((29,1))
    # # print (vector1)
    # PlotMatrix(vector1, Type, SaveName, save_dir)

    # vector2 = np.random.randn(29).reshape((29,1))
    # SaveName = "RandomVector2"
    # PlotMatrix(vector2, Type, SaveName, save_dir)

    # matrix = np.random.randn(29,10)
    # Type = "Random Matrix"
    # SaveName = "RandomMatrix"
    # PlotMatrix(matrix, Type, SaveName, save_dir)

# --------------------------------------------





