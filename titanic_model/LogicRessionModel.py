import preProcessData as pdata
import numpy as np
import pandas as pd
import math
import Cross
import matplotlib.pyplot as plt
Ntrain=10 #训练次数
iterator=30 #迭代次数
batch=50  #每次训练50个
publish=0.001 #惩罚率
eps=0.0001   #梯度范围

def training():
    X,Y,X_test=pdata.preProdata1()#通过数据处理方式一,得到训练集（包含验证集）和测试集
    thetas=[]#记录每次迭代更新的theta
    accuracys = []#记录每次验证的正确率
    #进行Ntrain次的训练
    for i in range(Ntrain):
        x_tain,y_train,x_test,y_test=Cross.simpleCross(X,Y)#简单交叉验证的方式得到训练和验证集
        #x_tain, y_train, x_test, y_test = k_cross(X, Y,i)
        start=0
        final_size=x_tain.shape[0]
        theta = np.random.rand(1, len(x_tain[0]))#每次训练随机初始化theta值
        errors=[]#记录一次训练中每次迭代的误差
        alpha =1
        #迭代
        for j in range(iterator):
            end=min(start+batch,final_size)
            error=y_train[start:end] - sigmod(theta, x_tain[start:end])
            ftheta=np.dot(error,x_tain[start:end])
            a=sum(abs(error[0]))
            errors.append(a)
            print('第{0}次训练,迭代了{1}次,误差为{2}'.format(i,j,a))
            #当梯度更新后的最大值小于eps时，认为已经得到了最优解
            if  max(abs(ftheta[0]))<eps:
                print('最大',i,ftheta[0])
                break
            #更新theta
            theta=theta+alpha*ftheta-theta*publish
            start=(start+batch)%final_size
            alpha = 4 / (1 + 5*j) + 0.0001#动态更新alpha

        thetas.append(theta)  #存取每次训练的theta

        #验证集进行验证
        y_pre = sigmod(theta, x_test)
        accuracys.append(Cross.get_rate(y_pre[0],y_test))
    #输出10次训练的所有准确率和平均准确率、最大准确率
    print(accuracys)
    print(sum(accuracys) / len(accuracys))
    max_accuracyIndex=accuracys.index(max(accuracys))
    print(accuracys[max_accuracyIndex])
    #对10次训练中拥有最大准确率的模型进行预测测试数据，并写入文件。
    y_pre = sigmod( thetas[max_accuracyIndex], np.array(X_test))
    y_pre=Cross.getResult(y_pre[0])
    Cross.getSave(y_pre)

def sigmod(theta,x):
    y=np.zeros([x.shape[0],1])
    for i in range(x.shape[0]):
        y[i]=1/(1+math.exp(-1*np.dot(theta,x[i].transpose())))
    return y.transpose()


training()