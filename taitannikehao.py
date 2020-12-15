# 数据分析包
import pandas as pd
import numpy as np
import random as rnd

#可视化包
import seaborn as sns
import matplotlib.pyplot as plt


#机器学习包
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#获取数据
train_df = pd.read_csv('C:/Users/月亮石/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/月亮石/Desktop/test.csv')

#combine变量将两数据集合并，以备后面用
combine = [train_df,test_df]

#验证地位高的成员存活率会高于地位低的
#先从训练数据中单独取出’Pclass’和’Survived’这两个特征数，然后根据’Pclass’特征来做分组,
#并就每组计算平均值(mean),然后就平均值做一个倒序. 
#最后可以看到地位高(Pclass=1)的成员存活率高很多(Survived>0.62)

train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()\
.sort_values(by='Survived',ascending=False)

#验证存活率和性别的关系
#首先单独选出“Sex”和“Survive的”然后根据“Sex”来分组，并就每组计算平均值mean,
#然后根据平均值排序（倒序）
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()\
.sort_values(by='Survived',ascending=False)

#Age可视化直方图
g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=20)

#Pclass可视化
grid = sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=3.8)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()

# Embarked和Sex用图表展示她们和Pclass与存活率的关联性
grid = sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep',\
        hue_order=["female","male"])
grid.add_legend()

#用seanbon画图，行（row）是Embarked，列（col）是Survived
grid = sns.FacetGrid(train_df,row='Embarked',col='Survived',size=2.2,aspect=1.6)
grid.map(sns.barplot,'Sex','Fare',alpha=.6,ci=None,order=['female','male'])
grid.add_legend()

#去掉Cabin、Ticket、Name、PassengerID
train_df = train_df.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
test_df  = test_df.drop(['Name','Ticket','Cabin'],axis=1)
combine = [train_df,test_df]
print(train_df.shape,test_df.shape)

#将female的值换成1，male换成0
if __name__ == '__main__':
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)
train_df.head()

# 处理缺失值和空值
#基于Pclass和Sex关联，处理Age，将其转换为5个连续的等级
grid = sns.FacetGrid(train_df,row='Pclass',col='Sex',size=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
plt.show()

#用中位数来填充Age的缺失值
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):

            guess_df = dataset[(dataset['Sex']==i)& \
                               (dataset['Pclass']==j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&\
                        (dataset.Sex == i)&(dataset.Pclass==j+1),'Age']\
            =guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
print(guess_ages)

#将Age切成5段看看与存活率的关系
train_df['AgeBand']=pd.cut(train_df['Age'],5) #切分
train_df[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean()\
.sort_values(by='AgeBand',ascending=True)
for dataset in combine:
    dataset.loc[dataset['Age']<=16,'Age'] = 0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age']=3
    dataset.loc[(dataset['Age']>64),'Age']=4
train_df.head(5)

# 删除AgeBand
train_df = train_df.drop(['AgeBand'],axis=1)
combine = [train_df,test_df]
train_df.head()

#将Parch和SibSp和Fare三个特征去掉
train_df = train_df.drop(['Parch','SibSp','Fare'],axis=1)
test_df = test_df.drop(['Parch','SibSp','Fare'],axis=1)
combine = [train_df,test_df]
train_df.head()

#处理Embarked特征
#用最长出现的值填充缺失值，然后在看关联性
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()\
.sort_values(by='Survived',ascending=False)

# 将Embarked转换成数字类型
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
train_df.head()
train_df.head()

#训练模型
X_train = train_df.drop('Survived',axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId',axis=1).copy()
print(X_train.shape,Y_train.shape,X_test.shape)

# 随机森林
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train,Y_train)*100,2)
print(acc_random_forest)
