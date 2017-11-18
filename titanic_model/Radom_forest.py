from sklearn.ensemble import RandomForestClassifier
import preProcessData as pdata
import Cross
import numpy as np
import pandas as pd
Ntrain=10
accuracys=[]
y_pre2=[]
X, Y, X_test = pdata.preProdata2()
for i in range(Ntrain):
    x_train, y_train, x_test, y_test = Cross.simpleCross(X, Y)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    y_pre = random_forest.predict(x_test)
    y_pre2.append(random_forest.predict(X_test))
    accuracys.append(Cross.get_rate(y_pre, y_test))

# 输出10次训练的所有准确率和平均准确率、最大准确率
print(accuracys)
print(sum(accuracys) / len(accuracys))
max_accuracyIndex = accuracys.index(max(accuracys))
print(accuracys[max_accuracyIndex])
# 对10次训练中拥有最大准确率的模型进行预测测试数据，并写入文件。
y_predict=Cross.getResult(y_pre2[max_accuracyIndex])
Cross.getSave(y_predict)