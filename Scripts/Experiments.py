import pickle
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import re,sys
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def kerasmodel():
    model = Sequential()
    model.add(Dense(256, input_dim=8, init='normal', activation='relu'))
    model.add(Dense(128,  init='normal', activation='relu'))
    model.add(Dense(64,  init='normal', activation='relu'))
    model.add(Dense(32, init='normal',activation='relu'))
    model.add(Dense(1, init='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def datapreparation_genderwise():
    #For male gender
    train_male = pd.read_csv('D:\\BLackFriday\\data\\train_new.csv')
    train_female = pd.read_csv('D:\\BLackFriday\\data\\train.csv')
    test_data=pd.read_csv('D:\\BLackFriday\\data\\test_new.csv')
    maley=train_female.ix[:,'Purchase']
    #femaley=train_female.ix[:,'Purchase']
    #print train_male.columns.values.tolist()
    #train_male=train_male.drop(['User_ID','Product_ID',"Purchase",],1)
    #train_female=train_female.drop(['User_ID','Product_ID',"Purchase","Gender"],1)
    #train_y=pd.concat([maley,femaley])

    print "Fitting male classifier"
    clf = xgb.XGBRegressor(n_estimators=1000)
    #clf=BaggingRegressor(base_estimator=est1,n_jobs=-1)
    clf.fit(train_male.values, maley.values)

    # est1 = xgb.XGBRegressor(n_estimators=1000)
    # #maleclf=KNeighborsRegressor()#n_estimators=1000,nthread=2)
    # #maleclf=KerasRegressor(build_fn=kerasmodel, nb_epoch=100, batch_size=100, verbose=1)
    # maleclf=BaggingRegressor(base_estimator=est1,n_jobs=-1)
    # maleclf.fit(train_male.values,maley.values)
    # print "Training female classifier"
    # est2=xgb.XGBRegressor(n_estimators=1000,nthread=2)
    # femaleclf=BaggingRegressor(base_estimator=est2,n_jobs=-1)
    # #femaleclf=KNeighborsRegressor()#KerasRegressor(build_fn=kerasmodel, nb_epoch=100, batch_size=100, verbose=1)
    # femaleclf.fit(train_female.values, femaley.values)

    #test_datacopy=test_data
    #test_data=test_data.drop(['User_ID','Product_ID'],1)
    ypred=[]
    for rows in test_data.values:
        if rows[0]=='M':
            rows=np.delete(rows,0)
            #print rows
            ypred.append(clf.predict(np.array([rows]).astype(np.float64))[0])
        elif rows[0] == 'F':
            rows=np.delete(rows, 0)
            #print rows
            ypred.append(clf.predict(np.array([rows]).astype(np.float64))[0])

    #sys.exit(0)

    print "Testing on both classifiers"
    submission = pd.DataFrame()
    submission['User_ID'] = test_data['User_ID']
    submission['Product_ID'] = test_data['Product_ID']
    submission['Purchase'] = ypred
    submission.to_csv('D:\\BLackFriday\\data\\Gender\\submission.csv',index=False)



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

if __name__=='__main__':
    datapreparation_genderwise()
    #datapreparation()