import sys,os
import pandas as pd

def merger():
    ff=open("D:\\BLackFriday\\data\\submission.csv",'r')
    content1=ff.readlines()

    f2=open("D:\\BLackFriday\\data\\submit.csv",'r')
    content2=f2.readlines()
    ypred=[]
    for line in content1:
        aa=line.split(",")[0]
        bb=line.split(",")[1].split("\n")[0]
        for lines in content2:
            a = lines.split(",")[0]
            b = lines.split(",")[1]
            c = lines.split(",")[2].split("\n")[0]
            if aa==a and bb==b:
                ypred.append(float(c))

    test_data = pd.read_csv('D:\\BLackFriday\\data\\test_new.csv')
    submission = pd.DataFrame()
    submission['User_ID'] = test_data['User_ID']
    submission['Product_ID'] = test_data['Product_ID']




    submission['Purchase'] = ypred
    submission.to_csv('D:\\BLackFriday\\data\\Gender\\submission_new.csv', index=False)



merger()