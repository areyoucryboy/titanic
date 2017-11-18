from sklearn import tree
import preProcessData as pdata
import Cross
import numpy as np
import pandas as pd
Ntrain=10
accuracys=[]
y_pre2=[]
X, Y, X_test = pdata.preProdata1()
for i in range(Ntrain):
    x_train, y_train, x_test, y_test = Cross.simpleCross(X, Y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    y_pre=clf.predict(x_test)
    y_pre2.append(clf.predict(X_test))
    accuracys.append(Cross.get_rate(y_pre, y_test))

print(accuracys)
print(sum(accuracys) / len(accuracys))
max_accuracyIndex = accuracys.index(max(accuracys))
print(accuracys[max_accuracyIndex])

y_predict = Cross.getResult(y_pre2[max_accuracyIndex])
Cross.getSave(y_predict)