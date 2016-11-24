import pickle
import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer,PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,train_test_split
from sklearn.feature_selection import RFECV
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

def runner():
    file=open('G:\\BLackFriday\\data\\trainFile','rb')
    X_Train,X_Valid,Y_Train,Y_valid,X_Test=pickle.load(file)
    # regr=AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=100)
    # regr=LinearRegression(n_jobs=-1)
    # regr=LogisticRegression()
    # polynomials=PolynomialFeatures(interaction_only=True)
    # polynomials.fit_transform(X_Train)
    # polynomials.fit_transform(X_Test)
    # polynomials.fit_transform(X_Valid)

    X_Test=list(X_Test)
    test=pd.read_csv('G:\\BLackFriday\\data\\test.csv')
    # for i in range(len(X_Test)):
    #     X_Test[i]=np.append(X_Test[i],[0,0])

    # print X_Test
    est=KNeighborsRegressor()#xgb.XGBRegressor(n_estimators=1500)
    #est2=GradientBoostingRegressor(n_estimators=100)
    #regr2=RFECV(estimator=est,)
    regr2=BaggingRegressor(base_estimator=est,n_jobs=-1)
    #regr2=RandomForestRegressor(n_jobs=-1,n_estimators=20)

    regr2.fit(X_Train,Y_Train)
    Y_valid_pred2=regr2.predict(X_Valid)
    print np.sqrt(mean_squared_error(Y_valid,Y_valid_pred2))
    # Y_Pred1=regr.predict(X_Test)
    Y_Pred2=regr2.predict(X_Test)
    #Y_Pred3=est2.predict(X_Test)
    # print regr.feature_importances_
    submission=pd.DataFrame()
    submission['User_ID']=test['User_ID']
    print submission.describe
    print submission.values
    submission['Product_ID']=test['Product_ID']
    submission['Purchase1']=Y_Pred2
    #submission['Purchase2']=Y_Pred3

    # submission['Purchase']=((Y_Pred1+Y_Pred2)/2)

    submission.to_csv('G:\\BLackFriday\\data\\submission.csv')

    # plt.plot([i for i in range(1,1001)],Y_Pred[0:1000],'b^',[i for i in range(1,1001)],Y_Test[0:1000],'rs')
    # plt.show()


    # for i in range(len(Y_Pred)):
    #     print Y_Pred[i],'-----',Y_Test[i]


    x=dict()
    # for i in len(train.values)
    # x.update({'age':[],'gender':[],'occ':[],'city':[],'stay':[],'marital':[],'prodcat1':[],'prodcat2':[],'prodcat3':[]})
    #
    # # x['']
    # x['age']=list(train['Age'].values)
    # x['gender']=list(train['Gender'].values)
    # x['occ']=list(train['Occupation'].values)
    # x['city']=list(train['City_Category'].values)
    # x['stay']=list(train['Stay_In_Current_City_Years'].values)
    # x['marital']=list(train['Marital_Status'].values)
    # x['prodcat1']=list(train['Product_Category_1'].values)
    # x['prodcat2']=list(train['Product_Category_2'].values)
    # x['prodcat3']=list(train['Product_Category_3'].values)
    # # for i in range(len(x['gender'])):
    # #
    # #     if x['gender'][i]=='F':
    # #         x['gender'][i]=0
    # #     else:
    # #         x['gender'][i]=1
    # #
    # # for dummy in range(10):
    # #     print type(x['age'][dummy])
    # #
    # # for i in range(len(x['age'])):
    # #
    # #     if x['age'][i]=='F':
    # #         x['gender'][i]=0
    # #     else:
    # #         x['gender'][i]=1
    # # print x['gender']
    #
    #
    #
    #
    # # print len(x['age'])
    # x=train.values
    # # for i in x:
    # #     for j in i:
    # #         if j==nan:

if __name__ =='__main__':
    runner()