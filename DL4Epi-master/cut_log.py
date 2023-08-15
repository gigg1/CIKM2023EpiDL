#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import glob
import numpy as np
import os

if __name__ == '__main__':

    # 工作文件夹，可以含有子文件夹
    text = "./log/cnnrnn_res/"                        #只需输入txt文件所在的文件夹
    dirs = os.listdir(text)

    for dir in dirs:
        directory = text + '/' + dir
        print (directory)
        if directory=="./log/cnnrnn_res//.DS_Store":
            continue

        # 打开文件
        with open(directory, "r") as f:
            # 读取文件
            data = f.readlines()
            # 获取txt文件的行数
            N = len(data)  
            # 打印行数
            # print(N)             
            # print (N)
            
            # print (cutStartIndex,cutEndIndex)
            # print (data[cutStartIndex:cutEndIndex])
            # print (data[0:cutStartIndex])
            # print (data)
            if N>0:
                line = data[-1]
            else:
                continue

            # invalid or NaN
            if not line.startswith('test rse') and N>2000:
                cutStartIndex=N-42-1
                cutEndIndex=N
                data=data[0:cutStartIndex]
                #print(data)

                #以写入的形式打开txt文件
                f = open(directory, "w")
                #将修改后的文本内容写入
                f.writelines(data)
                # 关闭文件
                f.close()
            else:
                continue
            
