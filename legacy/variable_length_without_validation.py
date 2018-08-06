import numpy as np
import wfdb as wf
import pandas as pd

table_path = 'table.csv'
ECG_folder_path = '/home/yanshuo/Desktop/research/dataset/'
ONE_HOT_ENCODE_LABEL = {'A':[0,0,0,1], '~':[0,0,1,0], 'N':[0,1,0,0], 'O':[1,0,0,0]}
dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)

trainD = {}
trainL = {}
validateD = {}
validateL = {}
testD = {}
testL = {}

def openData(filename):
    index = wf.rdsamp(ECG_folder_path + filename)
    record = index[0]    
    return record

def makeData(types, dataIndex, labelIndex, trainingPart):
    total = dataFromCSV.count(axis = 0)[types]
    trainPoint = int(total*trainingPart)+1

    for i in range(total):
        data = openData(dataFromCSV.iloc[i,dataIndex])
        label = dataFromCSV.iloc[i,labelIndex]
        if i < trainPoint:
            if data.shape in trainD:
                trainD[data.shape].append(data)
                trainL[data.shape].append(ONE_HOT_ENCODE_LABEL[label])
                
            else:
                trainD[data.shape] = [data]
                trainL[data.shape] = [ONE_HOT_ENCODE_LABEL[label]]
        else:
            if data.shape in testD:
                testD[data.shape].append(data)
                testL[data.shape].append(ONE_HOT_ENCODE_LABEL[label])
            else:
                testD[data.shape] = [data]
                testL[data.shape] = [ONE_HOT_ENCODE_LABEL[label]]

def main(trainingPart, testingPart):
    dataIndex = 0
    labelIndex = 1
    for i in range(1,8,2):
        makeData(i, dataIndex, labelIndex, trainingPart,)
        dataIndex += 2
        labelIndex += 2
    
    return trainD, trainL, testD, testL
