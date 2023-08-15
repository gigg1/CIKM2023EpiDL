#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import glob
import numpy as np
import pandas as pd

## Find the best performance in sets of logs
# extract the values from log
def extract_tst_from_log(filename):
    # empty file
    lines = open(filename).readlines()
    if len(lines) < 1:
        return 1e10, 1e10, -1,
    line = lines[-1]
    # invalid or NaN
    if not line.startswith('test rse'):
        return 1e10, 1e10, -1
    fields = line.split('|')
    tst_rse = float(fields[0].split()[2])
    tst_rae = float(fields[1].split()[2])
    tst_cor = float(fields[2].split()[2])

    # print (tst_rse,tst_rae,tst_cor)
    # return tst_rse, tst_cor
    return tst_rse, tst_rae, tst_cor

def format_logs(raw_expression):
    val_filenames = []
    BestModelFileName={}
    Results = {}
    for num in [1, 2, 4]:
        if num not in Results.keys():
            Results[num] = {}

        expressions = raw_expression.format(num)
        filenames = glob.glob(expressions)
        tuple_list = [extract_tst_from_log(filename) for filename in filenames]
        
        # print (tuple_list)

        if len(tuple_list) == 0:
            Results[num]["rmse"] = None
            Results[num]["rae"] = None
            Results[num]["corr"] = None
            continue
        # rse_list, cor_list = zip(*tuple_list)
        rse_list, rae_list, cor_list = zip(*tuple_list)
        index = np.argmin(rse_list)
        # print('horizon:{:2d}'.format(num), 'rmse: {:.4f}'.format(rse_list[index]), 'corr: {:.4f}'.format(cor_list[index]), 'best_model:', filenames[index])
        # print('horizon:{:2d}'.format(num), 'rmse: {:.4f}'.format(rse_list[index]), 'rae: {:.4f}'.format(rae_list[index]), 'corr: {:.4f}'.format(cor_list[index]), 'best_model:', filenames[index])
        
        Results[num]["rmse"] = '{:.4f}'.format(rse_list[index])
        Results[num]["rae"] = '{:.4f}'.format(rae_list[index])
        Results[num]["corr"] = '{:.4f}'.format(cor_list[index])

        BestModelFileName[num] = filenames[index]

    ResultsDf = pd.DataFrame.from_dict(Results)
    ResultsDf = ResultsDf[[1,2,4]]
    ResultsDf = ResultsDf.reindex(['rmse','rae','corr'])
    print (ResultsDf)

    return BestModelFileName

def Plot_ParameterAndPerformance(data):
    # --------------- parameters
    # horizon
    horizon_list=[1, 2, 4, 8]

    # hidden
    hidden_list=[5, 10, 20, 40, 50]

    # drop
    drop_list=[0.0, 0.2, 0.5]

    # window
    window_list=[2, 8, 32, 64, 128]

    # ratio
    ratio_list=[0.001, 0.01, 0.1, 1]

    # residual
    res_list=[4, 8, 16]

    # lambda
    lambda_list=[0.01, 0.1, 0.5]

    # raw_expression='./log/cnnrnn_res_epi/cnnrnn_res_epi.%s.hid-{}.drop-{}.w-{}.h-{}.ratio-{}.res-{}.lam-{}.out' % (data)
    # for horizon in horizon_list:
    #     for hidden in hidden_list:
    #         for drop in drop_list:
    #             for window in window_list:
    #                 for ratio in ratio_list:
    #                     for res in res_list:
    #                         for lam in lambda_list:
    #                             expressions = raw_expression.format(hidden,drop,window,horizon,ratio,res,lam)
    #                             filenames = glob.glob(expressions)
    #                             tuple_list = [extract_tst_from_log(filename) for filename in filenames]
    #                             if len(tuple_list) == 0:
    #                                 continue
    #                             rse_list, cor_list = zip(*tuple_list)
    #                             print ("Parameter",":",hidden,drop,window,horizon,ratio,res,lam)
    #                             print ("rse",":",rse_list)

    for horizon in horizon_list:
        print ("------------------------- Horizon", horizon)
        raw_expression='./log/cnnrnn_res_epi/cnnrnn_res_epi.%s.hid-{}.drop-*.w-*.h-{}.ratio-*.res-*.lam-*.out' % (data)
        for hidden in hidden_list:
            expressions = raw_expression.format(hidden,horizon)
            filenames = glob.glob(expressions)
            tuple_list = [extract_tst_from_log(filename) for filename in filenames]
            if len(tuple_list) == 0:
                continue
            rse_list, cor_list = zip(*tuple_list)
            rse_list_2=[]
            for value in rse_list:
                if value >= 10000000000.0:
                    continue
                else:
                    rse_list_2.append(value)

            averageValue=np.mean(rse_list_2)
            print ("Parameter",":",hidden)
            print ("rse",":",averageValue)

        # for drop in drop_list:
        
        # for window in window_list:
        
        # for ratio in ratio_list:
        
        # for res in res_list:
        
        # for lam in lambda_list:
                                
BestResultsFileDic={}
if __name__ == '__main__':

    for data in ['hhs']:
    # for data in ['hhs']:
        if data not in BestResultsFileDic.keys():
            BestResultsFileDic[data]={}
        print ("--------------------Data:",data)
        # ar
        print ("*" * 40)
        print ("AR","-",data)
        BestFileName = format_logs('./log/ar/ar.%s.d-*.w-*.h-{}.out' %(data))
        BestResultsFileDic[data]["AR"] = BestFileName

        # gar
        print('*' * 40)
        print ("GAR","-",data)
        BestFileName = format_logs('./log/gar/gar.%s.d-*.w-*.h-{}.out' %(data))
        BestResultsFileDic[data]["GAR"] = BestFileName

        # var
        print('*' * 40)
        print ("VAR","-",data)
        BestFileName = format_logs('./log/var/var.%s.d-*.w-*.h-{}.out' %(data))
        BestResultsFileDic[data]["VAR"] = BestFileName

        # cnnrnn_res
        print('*' * 40)
        print ("CNNRNN-Res","-",data)
        BestFileName = format_logs('./log/cnnrnn_res/cnnrnn_res.%s.hid-*.drop-*.w-*.h-{}.ratio-*.res-*.out' % (data))
        BestResultsFileDic[data]["CNNRNN-Res"] = BestFileName

        print ("")
        print ("")
        # Plot_ParameterAndPerformance(data)

        # # cnnrnn
        # format_logs('./log/cnnrnn/cnnrnn.%s.hid-*.drop-*.w-*.h-{}.out' %(data))
        # print('*' * 40)
        # # rnn
        # format_logs('./log/rnn/rnn.%s.hid-*.drop-*.w-*.h-{}.out' %(data))
        # print('*' * 40)

        # # var_mask
        # format_logs('./log/var_mask/var_mask.%s.d-*.w-*.h-{}.out' %(data))
        # print('*' * 40)

    # print ("--------------- Best model file names:")
    # for dataset in BestResultsFileDic.keys():
    #     for model in BestResultsFileDic[dataset].keys():
    #         for hor in BestResultsFileDic[dataset][model]:
    #             print (dataset,"-",model,"-",hor,":")
    #             print (BestResultsFileDic[dataset][model][hor])
    #             print (" ")
            

