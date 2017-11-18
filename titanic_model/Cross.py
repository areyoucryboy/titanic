import numpy as np
import pandas as pd
import sys
import os
percent_train=0.9 #从训练集中选取0.9作为训练，剩下0.1作为验证集
PHR=10  #k折交的k=10
#简单交叉验证
def simpleCross(x,y):
    lens=x.shape[0]
    index_array = np.arange(lens)
    np.random.shuffle(index_array)
    x_train=[]
    y_train = []
    for i in index_array:
        x_train.append(x.loc[i])
        y_train.append(y.loc[i])
    train_location=int(lens * percent_train)
    return np.array(x_train[:train_location]),np.array(y_train[:train_location]),\
           np.array(x_train[train_location:]),np.array(y_train[train_location:])
#k-折交叉验证
def k_cross(x,y,i):
    num=x.shape[0]/PHR
    x_test=x.loc[i*num:num*(i+1)-1]
    x_train=x.drop(np.arange(i*num,num*(i+1)-1))
    y_test = y.loc[i * num:num * (i + 1) - 1]
    y_train =y.drop(np.arange(i *num, num * (i + 1) - 1))
    return np.array(x_train),np.array(y_train),\
           np.array(x_test),np.array(y_test)

#得到正确率
def get_rate(y_pre,y_test):
    y_p=[]
    for i in range(len(y_pre)):
        if y_pre[i]>=0.5:
            y_p.append(1)
        else:
            y_p.append(0)
    #print(y_test)
    return (len(y_test)-sum(abs(y_p-y_test)))/len(y_test)
#将预测结果变为0,1的形式
def getResult(y_pre):
    y_p = []
    for i in range(len(y_pre)):
        if y_pre[i] >= 0.5:
            y_p.append(1)
        else:
            y_p.append(0)
    return y_p
#将预测结果写入到csv文件中
def getSave(y_pre):
    save=pd.DataFrame({'PassengerId':892+np.arange(len(y_pre)),'Survived':y_pre})
    save.to_csv(os.path.join(sys.path[0],'result.csv')
    print('保存成功')
