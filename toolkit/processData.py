from .openFile import openData
import pandas as pd
import numpy as np

ECG_folder_path = '/home/hsiehch/dataset/'
newData = []
newLabel = []
ONE_HOT_ENCODE_LABEL = {'A':0, '~':1, 'N':2, 'O':3}
CLASS_AMOUNT = 0
dataRemainder = 0

def makeData(totalDataInThisClass, dataIndex, labelIndex, table, desired_data_point, overlap_dot):
    
    global CLASS_AMOUNT
    global newData
    global newLabel
    global dataRemainder
    
    for i in range(totalDataInThisClass):
        dataLen = len(openData(ECG_folder_path, table.iloc[i,dataIndex]))
        
        dataRemainder = 0
        
        if(dataLen >= desired_data_point):
            reduceData(i, dataIndex, labelIndex, table, desired_data_point, overlap_dot)
        
        if(dataRemainder != 0 ):
            makeInsuffitientData(i, dataIndex, labelIndex, table, desired_data_point)
            
        if(dataLen < desired_data_point):
            increaseData(i, dataIndex, labelIndex, table, desired_data_point)
        
        
    print(str(table.iloc[i,labelIndex]) + ": " + str(CLASS_AMOUNT))
    
    
    newData = np.array(newData)
    newLabel = np.array(newLabel)
    
    return CLASS_AMOUNT, newData, newLabel

def reduceData(i ,dataIndex, labelIndex, table, desired_data_point, overlap_dot):
    j = 1
    previous_j = 0
    data = openData(ECG_folder_path, table.iloc[i,dataIndex])
    label = table.iloc[i,labelIndex]
    global CLASS_AMOUNT
    global newData
    global newLabel
    global dataRemainder
    
    while j <= len(data):
        dataRemainder += 1
        if dataRemainder == desired_data_point:
            CLASS_AMOUNT += 1
            newData.append(data[previous_j : j])
            j = j - overlap_dot
            previous_j = j
            newLabel.append(ONE_HOT_ENCODE_LABEL[label])
            dataRemainder = 0
        j += 1

def makeInsuffitientData(i, dataIndex, labelIndex, table, desired_data_point):
    global CLASS_AMOUNT
    global newData
    global newLabel
    data = openData(ECG_folder_path, table.iloc[i,dataIndex])
    label = table.iloc[i,labelIndex]
    newData.append(data[ : desired_data_point])
    newLabel.append(ONE_HOT_ENCODE_LABEL[label])
    CLASS_AMOUNT += 1

def increaseData(i, dataIndex, labelIndex, table, desired_data_point):
    global CLASS_AMOUNT
    global newData
    global newLabel
    data = openData(ECG_folder_path, table.iloc[i,dataIndex])
    label = table.iloc[i,labelIndex]
    tmp = data
    numOfBatch = desired_data_point // len(data)
    leftData = desired_data_point % len(data)
    
    print(numOfBatch)
    print(leftData)
    print('======')
    while numOfBatch > 1 :
        tmp = np.append(tmp, data)
        numOfBatch -= 1
    if(leftData != 0):
        tmp = np.append(tmp, data[ : leftData ])
    newData.append(tmp)
    newLabel.append(ONE_HOT_ENCODE_LABEL[label])
    CLASS_AMOUNT += 1
