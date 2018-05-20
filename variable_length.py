import numpy as np
import wfdb as wf
import pandas as pd

table_path = 'table.csv'
ECG_folder_path = '/home/yanshuo/Desktop/research/dataset/'
ONE_HOT_ENCODE_LABEL = {'A':[0,0,0,1], '~':[0,0,1,0], 'N':[0,1,0,0], 'O':[1,0,0,0]}
dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)

# afTotal = dataFromCSV.count(axis = 0)[3]
# noiseTotal = dataFromCSV.count(axis = 0)[1]
# otherTotal = dataFromCSV.count(axis = 0)[5]
# normalTotal = dataFromCSV.count(axis = 0)[7]

# dataIndex = 2
# labelIndex = 3
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

def makeData(types, dataIndex, labelIndex, trainingPart, validationPart):
    total = dataFromCSV.count(axis = 0)[types]
    trainPoint = int(total*trainingPart)+1
    validatePoint = trainPoint + int(total*validationPart)+1

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
        elif i < validatePoint:
            if data.shape in validateD:
                validateD[data.shape].append(data)
                validateL[data.shape].append(ONE_HOT_ENCODE_LABEL[label])
            else:
                validateD[data.shape] = [data]
                validateL[data.shape] = [ONE_HOT_ENCODE_LABEL[label]]
        else:
            if data.shape in testD:
                testD[data.shape].append(data)
                testL[data.shape].append(ONE_HOT_ENCODE_LABEL[label])
            else:
                testD[data.shape] = [data]
                testL[data.shape] = [ONE_HOT_ENCODE_LABEL[label]]

    


def main(trainingPart, validationPart, testingPart):
    dataIndex = 0
    labelIndex = 1
    for i in range(1,8,2):
        makeData(i, dataIndex, labelIndex, trainingPart, validationPart)
        dataIndex += 2
        labelIndex += 2

    return trainD, trainL, validateD, validateL, testD, testL

    
# main(0.5, 0.2, 0.3)
# print(testD)


