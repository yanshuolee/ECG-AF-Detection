import numpy as py
import wfdb as wf
import pandas as pd
def chang(persan):
    # -- coding: utf-8 --
    TEXTN=0
    TEXTA=0
    TEXTO=0
    TEXTZZ=0
    ttt=0
    exceldata = pd.read_csv('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\revised_REFERENCE-v3.csv', header=None)
    testA = exceldata.iloc[0,1]
    record = wf.rdsamp('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\A00001')
    trainingA = record.p_signals[0:, 0]
    for i ,row in exceldata.iterrows():
        
        if (exceldata.iloc[i,1]=='A' and i>1):
            TEXTA=TEXTA+1
        if (exceldata.iloc[i,1]=='N' and i>1):
            TEXTN=TEXTN+1
        if (exceldata.iloc[i,1]=='O' and i>1):
            TEXTO=TEXTO+1
        if (exceldata.iloc[i,1]=='~' and i>1):
            TEXTZZ=TEXTZZ+1
    newTEXTN=TEXTN*(persan/100)+1
    newTEXTA=TEXTA*(persan/100)+1
    newTEXTO=TEXTO*(persan/100)+1
    newTEXTZZ=TEXTZZ*(persan/100)+1
    exceldata = pd.read_csv('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\revised_REFERENCE-v3.csv', header=None)
    for i ,row in exceldata.iterrows():
        print(i)
        if(i+1<10):
            record = wf.rdsamp('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\A0000'+"%d"%(i+1))
        elif(i+1<100):
            record = wf.rdsamp('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\A000'+"%d"%(i+1))
        elif(i+1<1000):
            record = wf.rdsamp('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\A00'+"%d"%(i+1))
        elif(i+1<10000):
            record = wf.rdsamp('C:\\Users\\進化中\\Desktop\\evolution\\training2017\\A0'+"%d"%(i+1))
        data = record.p_signals[0:, 0]
        if (exceldata.iloc[i,1]=='N'and i>0):
            if (newTEXTN>0):
                newTEXTN=newTEXTN-1
                trainingA=py.vstack((trainingA,data))
                testA=py.vstack((testA,exceldata.iloc[i,1]))
                
            elif(ttt==0):
                trainingB=data
                testB=exceldata.iloc[i,1]
                ttt=ttt+1
            else :
                trainingB=py.vstack((trainingB,data))
                testB=py.vstack((testB,exceldata.iloc[i,1]))
                
        if (exceldata.iloc[i,1]=='A'and i>0):
            if (newTEXTA>0):
                newTEXTA=newTEXTA-1
                trainingA=py.vstack((trainingA,data))
                testA=py.vstack((testA,exceldata.iloc[i,1]))
            elif(ttt==0):
                trainingB=data
                testB=exceldata.iloc[i,1]
                ttt=ttt+1
            else :
                trainingB=py.vstack((trainingB,data))
                testB=py.vstack((testB,exceldata.iloc[i,1]))
                
        if (exceldata.iloc[i,1]=='O'and i>0):
            if (newTEXTO>0):
                newTEXTO=newTEXTO-1
                trainingA=py.vstack((trainingA,data))
                testA=py.vstack((testA,exceldata.iloc[i,1]))
            elif(ttt==0):
                trainingB=data
                testB=exceldata.iloc[i,1]
                ttt=ttt+1
            else :
                trainingB=py.vstack((trainingB,data))
                testB=py.vstack((testB,exceldata.iloc[i,1]))
                
        if (exceldata.iloc[i,1]=='~'and i>0):
            if (newTEXTZZ>0):
                newTEXTZZ=newTEXTZZ-1
                trainingA=py.vstack((trainingA,data))
                testA=py.vstack((testA,exceldata.iloc[i,1]))
            elif(ttt==0):
                trainingB=data
                testB=exceldata.iloc[i,1]
                ttt=ttt+1
            else :
                trainingB=py.vstack((trainingB,data))
                testB=py.vstack((testB,exceldata.iloc[i,1]))
                
        print(trainingA)
        print(testA)    
    print(trainingB)
    print(testB)

chang(70)
