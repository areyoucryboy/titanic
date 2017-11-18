import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns

def get_person(passage):
    age,sex=passage
    return 'child' if age>16 else sex

def IS_two(Family):
    y=[]
    for i in Family.values:
        if i>1 and i<5:
            y.append(True)
        else:
            y.append(False)
    return y

#数据预处理方式二
def preProdata2():
    #读取训练样本和需预测的样本
    titanic_df = pd.read_csv(os.path.join(sys.path[0],'titanic_train.csv'))
    test_df=pd.read_csv(os.path.join(sys.path[0],'test.csv'))
    #观察整体数据情况
    titanic_df.head()
    titanic_df.info()
    test_df.head()
    test_df.info()


    #####对年龄进行处理



    #fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    #axis1.set_title('Original Age value')
    #axis2.set_title('new Age Value')
    # 获取均值，方差和缺失值的数量
    average_age_titanic = titanic_df["Age"].mean()
    std_age_titanic = titanic_df["Age"].std()
    count_nan_age_titanic = titanic_df["Age"].isnull().sum()
    # 然后随机生成该范围内的值
    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,
                               size=count_nan_age_titanic)
    # 扔掉所有的空值，然后画出年龄的各区间频数分布图
    #titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
    # 进行缺失值替换
    titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
    # 画出进行缺失值替换之后的频数直方图
    #titanic_df['Age'].hist(bins=70, ax=axis2)

    #同样对于需要预测的测试集进行填充
    average_age_test = test_df["Age"].mean()
    std_age_test = test_df["Age"].std()
    count_nan_age_test = test_df["Age"].isnull().sum()
    # 然后随机生成该范围内的值
    rand_1 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test,
                               size=count_nan_age_test)
    # 进行缺失值替换
    test_df["Age"][np.isnan(test_df["Age"])] = rand_1



    #####对于年龄和性别的相关性

    #对于各年龄段的存活率的研究,由下图可得16岁一下的小孩的存活率略高一下。
    #fig, axis1 = plt.subplots(1,1, figsize=(20, 5))
    Age_avgSur = titanic_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
   # sns.barplot(x='Age', y='Survived', data=Age_avgSur, order=np.arange(80), ax=axis1)

    #对于性别的影响,由下图明显女性的存活率更高
    #fig, axis2 = plt.subplots(1,1, figsize=(10, 5))
    Sex_avgSur = titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
   # sns.barplot(x='Sex', y='Survived', data=Sex_avgSur, order=['male','female'], ax=axis2)

    titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 0
    titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 1

    test_df.loc[test_df["Sex"] == "female", "Sex"] = 0
    test_df.loc[test_df["Sex"] == "male", "Sex"] = 1
    '''
    #所以综合这两个因素，对于16岁一下的人不区别性别,进行onehot编码，同时去掉male这一列
    titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)
    titanic_df.drop(['Sex'], axis=1, inplace=True)
    person_dummies_titanic = pd.get_dummies(titanic_df['Person'])
    person_dummies_titanic.columns = ['Child', 'Female', 'Male']
    person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
    titanic_df = titanic_df.join(person_dummies_titanic)
    titanic_df.drop(['Person'],axis=1, inplace=True)

    test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)
    test_df.drop(['Sex'], axis=1, inplace=True)
    person_dummies_test = pd.get_dummies(test_df['Person'])
    person_dummies_test.columns = ['Child', 'Female', 'Male']
    person_dummies_test.drop(['Male'], axis=1, inplace=True)
    test_df = test_df.join(person_dummies_test)
    test_df.drop(['Person'], axis=1, inplace=True)
    '''

    ####将兄弟姐妹和父母放一起考虑

    titanic_df['Family'] = titanic_df["Parch"] + titanic_df["SibSp"] + 1
    #由图可看出2,3,4的存活率更高，独自一人的较小，同时人数多于4个存活率也不多，于是分为三类
    #fig, axis3 = plt.subplots(1, 1, figsize=(10, 5))
    Family_avgSur = titanic_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
   # sns.barplot(x='Family', y='Survived', data=Family_avgSur, order=np.arange(13)+1, ax=axis3)


    titanic_df['Family'][titanic_df['Family'] == 1] =1
    y = IS_two(titanic_df['Family'])
    titanic_df['Family'][y] = 3
    titanic_df['Family'][titanic_df['Family'] >= 5] = 2
    person_dummies_titanic = pd.get_dummies(titanic_df['Family'])
    person_dummies_titanic.columns = ['alone', 'Family1','Family2']
    #person_dummies_titanic.drop(['Family2'], axis=1, inplace=True)
    titanic_df = titanic_df.join(person_dummies_titanic)
    titanic_df.drop(['Family'],axis=1, inplace=True)
    titanic_df.drop(['Parch'], axis=1, inplace=True)
    titanic_df.drop(['SibSp'], axis=1, inplace=True)

    test_df['Family'] = test_df["Parch"] + test_df["SibSp"] + 1
    test_df['Family'][test_df['Family'] == 1] =1
    y = IS_two(test_df['Family'])
    test_df['Family'][y] =3
    test_df['Family'][test_df['Family'] >= 5] = 2
    person_dummies_test= pd.get_dummies(test_df['Family'])
    person_dummies_test.columns = ['alone', 'Family1','Family2']
    #person_dummies_test.drop(['Family2'], axis=1, inplace=True)
    test_df = test_df.join(person_dummies_test)
    test_df.drop(['Family'],axis=1, inplace=True)
    test_df.drop(['Parch'], axis=1, inplace=True)
    test_df.drop(['SibSp'], axis=1, inplace=True)

    fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(x='Pclass', data=titanic_df, ax=axis1)  # 在第一个子图上
    Pclass_survived=titanic_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()
    sns.barplot(x='Pclass', y='Survived', data=Pclass_survived, order=[1,2,3], ax=axis2)
    #plt.show()


    #对Pclass进行onehot编码

    person_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])
    person_dummies_titanic.columns = ['Pclass0', 'Pclass1', 'Pclass2']
    person_dummies_titanic.drop(['Pclass2'], axis=1, inplace=True)
    titanic_df = titanic_df.join(person_dummies_titanic)
    titanic_df.drop(['Pclass'],axis=1, inplace=True)

    person_dummies_test= pd.get_dummies(test_df['Pclass'])
    person_dummies_test.columns = ['Pclass0', 'Pclass1', 'Pclass2']
    person_dummies_test.drop(['Pclass2'], axis=1, inplace=True)
    test_df = test_df.join(person_dummies_test)
    test_df.drop(['Pclass'],axis=1, inplace=True)




    #去除Cabin,Name,Ticket,PassageId四列
    titanic_df.drop(['Cabin','Name','Ticket','PassengerId'], axis=1, inplace=True)
    test_df.drop(['Cabin','Name','Ticket','PassengerId'], axis=1, inplace=True)

    #titanic_df['PassengerId']=1
    #test_df['PassengerId']=1


    #替换Embarked,并onehot编码

    fig, axis1 = plt.subplots(1,1, figsize=(10, 5))  # 创建一个画布有是三个子图
    #sns.countplot(x='Embarked', data=titanic_df, ax=axis1)  # 在第一个子图上画出以s,c,q为x轴,他们的数量为y轴的图形。
    #由图可知S的数量最多,用s来替换缺失值
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')  # because 's' is more
    #Embarked_survived=titanic_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
    #sns.barplot(x='Embarked', y='Survived', data=Embarked_survived, order=['S','C','Q'], ax=axis1)

    titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0
    titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 1
    titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 2

    test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0
    test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 1
    test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 2
    #plt.show()

    embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])  # 进行one-hot编码
    embark_dummies_titanic.columns=['S','C','Q']
    embark_dummies_titanic.drop(['S'], axis=1, inplace=True)  # 然后丢弃s那一列
    titanic_df = titanic_df.join(embark_dummies_titanic)  # 把c和q两列连接到训练数据中去
    titanic_df.drop(['Embarked'],axis=1, inplace=True)

    embark_dummies_test = pd.get_dummies(test_df['Embarked'])  # 进行one-hot编码
    embark_dummies_test.columns = ['S', 'C', 'Q']
    embark_dummies_test.drop(['S'], axis=1, inplace=True)  # 然后丢弃s那一列
    test_df = test_df.join(embark_dummies_titanic)  # 把c和q两列连接到训练数据中去
    test_df.drop(['Embarked'], axis=1, inplace=True)



    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

    simplifyData(titanic_df,test_df)
    y=titanic_df['Survived']
    titanic_df.drop(['Survived'], axis=1, inplace=True)

    return titanic_df,y,test_df

#分别对训练集和测试集中的Age和Fare两列进行数据归一化
def simplifyData(titanic_df,test_df):

    min_value=min(titanic_df['Fare'])
    max_value=max(titanic_df['Fare'])
    titanic_df['Fare']=(titanic_df['Fare']-min_value)/(max_value-min_value)
    min_value2 = min(test_df['Fare'])
    max_value2 = max(test_df['Fare'])
    test_df['Fare'] = (test_df['Fare'] - min_value) / (max_value - min_value)

    min_value3 = min(titanic_df['Age'])
    max_value3 = max(titanic_df['Age'])
    titanic_df['Age'] = (titanic_df['Age'] - min_value) / (max_value - min_value)
    min_value4 = min(test_df['Age'])
    max_value4 = max(test_df['Age'])
    test_df['Age'] = (test_df['Age'] - min_value) / (max_value - min_value)


#数据预处理方式一
def preProdata1():
    #读取
    titanic_df = pd.read_csv(os.path.join(sys.path[0],'titanic_train.csv'))
    test_df=pd.read_csv(os.path.join(sys.path[0],'test.csv'))

    #利用中位数进行填充Age的缺失值
    titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)

    test_df["Age"].fillna(test_df["Age"].median(), inplace=True)

    #删除一下这几列
    titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1,inplace=True)
    test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1,inplace=True)

    #对sex中的性别进行替换
    titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0
    titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1

    test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
    test_df.loc[test_df["Sex"] == "female", "Sex"] = 1

    #替换Embarked
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')  # because 's' is more
    titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0
    titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 1
    titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 2

    test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0
    test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1
    test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2

    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

    y=titanic_df['Survived']

    titanic_df.drop(['Survived'],axis=1,inplace=True)
    simplifyData(titanic_df,test_df)
    print(titanic_df)
    return titanic_df,y,test_df

#数据预处理方式三
