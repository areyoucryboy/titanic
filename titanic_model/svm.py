from sklearn.svm import SVC, LinearSVC
import preProcessData as pdata
import Cross
Ntrain=10
X, Y, X_test = pdata.preProdata2()
accuracys=[]
y_pre2=[]
for i in range(Ntrain):
    x_tain, y_train, x_test, y_test = Cross.simpleCross(X, Y)
    clf= SVC(C=1, kernel='rbf', degree=3, gamma='auto', coef0=0.0)
    clf.fit(x_tain, y_train)
    y_pre = clf.predict(x_test)
    accuracys.append(Cross.get_rate(y_pre, y_test))
    y_pre2.append(clf.predict(X_test))

print(accuracys)
print(sum(accuracys) / len(accuracys))
max_accuracyIndex = accuracys.index(max(accuracys))
print(accuracys[max_accuracyIndex])
y_predict=Cross.getResult(y_pre2[max_accuracyIndex])
Cross.getSave(y_predict)