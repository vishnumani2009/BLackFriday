import pickle
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

def datapreparation_genderwise():
    #For male gender
    train_male = pd.read_csv('G:\\BLackFriday\\data\\Gender\\Male.csv')
    train_female = pd.read_csv('G:\\BLackFriday\\data\\Gender\\Female.csv')
    train_male=train_male.drop(['User_ID','Product_ID','Gender'])
    train_female=train_male.drop(['User_ID','Product_ID','Gender'])




def datapreparation():
    train=pd.read_csv('G:\\BLackFriday\\data\\train.csv')
    test=pd.read_csv('G:\\BLackFriday\\data\\test.csv')
    train.Age[train['Age']=='0-17']=1
    train.Age[train['Age']=='18-25']=2
    train.Age[train['Age']=='26-35']=3
    train.Age[train['Age']=='36-45']=4
    train.Age[train['Age']=='46-50']=5
    train.Age[train['Age']=='51-55']=6
    train.Age[train['Age']=='55+']=7
    train.Gender[train['Gender']]

    test.Age[test['Age']=='0-17']=1
    test.Age[test['Age']=='18-25']=2
    test.Age[test['Age']=='26-35']=3
    test.Age[test['Age']=='36-45']=4
    test.Age[test['Age']=='46-50']=5
    test.Age[test['Age']=='51-55']=6
    test.Age[test['Age']=='55+']=7
    train.Product_Category_1[pd.isnull(train['Product_Category_1'])]=0
    test.Product_Category_1[pd.isnull(test['Product_Category_1'])]=0
    train.Product_Category_2[pd.isnull(train['Product_Category_2'])]=0
    test.Product_Category_2[pd.isnull(test['Product_Category_2'])]=0
    train.Product_Category_3[pd.isnull(train['Product_Category_3'])]=0
    test.Product_Category_3[pd.isnull(test['Product_Category_3'])]=0

    print train.values
    list_y=train['Purchase'].values
    train=train.drop('Purchase',axis=1)
    # train=encode_onehot(df=train,cols=['Age','Gender','City_Category','Stay_In_Current_City_Years','Marital_Status'])
    for i in ['Gender','City_Category','Stay_In_Current_City_Years']:
        oneHot=pd.get_dummies(train[i] )
        oneHot1=pd.get_dummies(test[i])
        # oneHot.columns=train.columns
        # oneHot.index=train.index
        train=train.drop(i,axis=1)
        test=test.drop(i,axis=1)
        train[oneHot.columns]=oneHot
        test[oneHot1.columns]=oneHot1
    print train.columns

    count=0
    print len(train.values)
    list_x=train.values
    list_x_test=test.values
    columns=train.columns
    index=train.index
    for i in list_x:
        # print i[1]
        list_i=i[1].split('P')
        i[1]= int(list_i[1])
    for i in list_x_test:
        # print i[1]
        list_i=i[1].split('P')
        i[1]= int(list_i[1])
    print len(list_y)
    print len(list_x),len(list_y)
    normalizer=Normalizer()
    list_x_test=normalizer.fit_transform(list_x_test)
    list_x=normalizer.fit_transform(list_x)
    file=open('G:\\BLackFriday\\data\\trainFile','wb')
    X_Train,X_Test,Y_Train,Y_Test=train_test_split(list_x,list_y,test_size=0.3)
    X_valid=X_Test
    Y_valid=Y_Test
    pickle.dump([X_Train,X_valid,Y_Train,Y_valid,list_x_test],file)
    file.close()


def merger():
    ff=open()