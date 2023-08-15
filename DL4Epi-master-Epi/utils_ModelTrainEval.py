import torch
import torch.nn as nn

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
import numpy as np

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    # print ("--------- Evaluate")
    counter = 0
    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        if modelName == "CNNRNN_Res_epi":
            output, EpiOutput, _, _, _ = model(X);
        else:
            output = model(X);

        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict, output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)

        counter = counter + 1
        
        if torch.__version__ < '0.4.0':
            total_loss += evaluateL2(output * scale , Y * scale).data[0]
            total_loss_l1 += evaluateL1(output * scale , Y * scale).data[0]
        else:
            total_loss += evaluateL2(output * scale , Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale , Y * scale).item()

        n_samples += (output.size(0) * loader.m);

        rmselist=[]
        outputValue = (output * scale).T
        RealValue = (Y * scale).T
        maelist=[]
        for i in range(0,len(outputValue)):
            LengthOfTime=len(outputValue[i])
            # print (math.sqrt(evaluateL2(outputValue[i] , RealValue[i] ).item()/LengthOfTime))
            rmselist.append(math.sqrt(evaluateL2(outputValue[i], RealValue[i]).item()/LengthOfTime))

    # Why divide loader.rse and loader.rae here
    # print (loader.rse)
    # print (loader.rae)
    # rse = math.sqrt(total_loss / n_samples)/loader.rse
    # rae = (total_loss_l1/n_samples)/loader.rae

    rse = math.sqrt(total_loss / n_samples)
    rae = (total_loss_l1/n_samples)

    predict = predict.data.numpy();
    Ytest = test.data.numpy();

    # tmp = 0
    # scale = loader.scale.data.numpy()[0]
    # for loc in range(0, predict.shape[1]):
    #     tmp = tmp + mean_squared_error(Ytest[:,loc]*scale, predict[:,loc]*scale)
    # tmp1 = np.sqrt(tmp/predict.shape[1])
    # print ("tmp1", tmp1)
    # print ("rse", rse)

    correlation = 0;
    # predict & Ytest : (test time steps, locations) 
    # sigma_p & sigma_g & mean_p & mean_g : (locations, ) calculate the average value of each column
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    # root-mean-square error, absolute error, correlation
    return rse, rae, correlation;

def train(loader, data, model, criterion, optim, batch_size, modelName, Lambda):
    model.train();
    total_loss = 0;
    n_samples = 0;
    counter = 0

    # print ("--------- Train")

    for inputs in loader.get_batches(data, batch_size, True):
        counter += 1
        X, Y = inputs[0], inputs[1]

        model.zero_grad();

        if modelName == "CNNRNN_Res_epi":
            output, EpiOutput, _, _, _ = model(X);
        else:
            output = model(X);

        scale = loader.scale.expand(output.size(0), loader.m)
        # loss = criterion(output * scale, Y * scale);
        
        # --------------------------------------------
        # modified loss with epidemiological constrains
        if modelName == "CNNRNN_Res_epi":
            loss = criterion(output * scale, Y * scale) + Lambda*criterion(EpiOutput * scale, Y * scale);
        else:
            loss = criterion(output * scale, Y * scale);

        # print ("Count",counter)
        # print ("Y:", Y.shape)
        # print ("X:", X.shape)
        # print ("X[:,-1,:]", X[:,-1,:].shape)

        # --------------------------------------------
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True);
        optim.step();

        # --------------------------------------------
        # ------------ set the parameter constraints
        # print ("**********************")
        # print (model.parameters())

        # if modelName == "CNNRNN_Res_epi":
        #     for name, para in model.named_parameters():
        #         # print (name, ":",para.size())
        #         if name == "Beta" or name == "Gamma" or name == "Mu":
        #             para.data.clamp_(min=0, max=1)
        #         if name == "mask_mat":
        #             para.data.clamp_(min=0, max=None)
        #             # para.data.clamp_(min=0, max=1)

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

def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval();
    Y_predict = None;
    Y_true = None;
    X_true = None

    # print ("--------- Get prediction")
    counter = 0
    if modelName == "CNNRNN_Res_epi":
        BetaList = None
        GammaList = None
        NGMList = None

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        
        if modelName == "CNNRNN_Res_epi":
            output, EpiOutput, Beta, Gamma, NGMT = model(X);
        else:
            output = model(X);
        
        counter = counter+1

        if Y_predict is None:
            Y_predict = output.cpu()
            Y_true = Y.cpu()
            X_true = X.cpu()

            BetaList = Beta.cpu()
            GammaList = Gamma.cpu()
            NGMList = NGMT.cpu()
        else:
            Y_predict = torch.cat((Y_predict,output.cpu()))
            Y_true = torch.cat((Y_true, Y.cpu()))
            X_true = torch.cat((X_true, X.cpu()))

            if modelName == "CNNRNN_Res_epi":
                BetaList = torch.cat((BetaList, Beta.cpu()))
                GammaList = torch.cat((GammaList, Gamma.cpu()))
                NGMList = torch.cat((NGMList, NGMT.cpu()))

        # scale = loader.scale.expand(output.size(0), loader.m)

    scale = loader.scale
    # print (X_true.shape)
    # print (scale)

    # Time * location
    Y_predict = (Y_predict * scale)
    Y_true = (Y_true * scale)
    X_true = (X_true * scale)
    
    Y_predict = Y_predict.detach().numpy()
    Y_true = Y_true.detach().numpy()
    X_true = X_true.detach().numpy()

    if modelName == "CNNRNN_Res_epi":
        BetaList = BetaList.detach().numpy()
        GammaList = GammaList.detach().numpy()
        NGMList = NGMList.detach().numpy()

    # print (BetaList.shape)
    # print (GammaList.shape)
    # print (NGMList.shape)
    if modelName == "CNNRNN_Res_epi":
        return X_true, Y_predict, Y_true, BetaList, GammaList, NGMList
    else:
        return X_true, Y_predict, Y_true
