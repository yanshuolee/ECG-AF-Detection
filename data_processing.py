import numpy as np
import wfdb as wf
import pandas as pd

table_path = 'table.csv'
ECG_folder_path = '/home/hsiehch/dataset/'

class generateData():
    
    newData = []
    newLabel = []
    ONE_HOT_ENCODE_LABEL = {'A':[0,0,0,1], '~':[0,0,1,0], 'N':[0,1,0,0], 'O':[1,0,0,0]}

    def __init__(self, seconds, overlap_dot = 0):
        SAMPLE_RATE = 300
        self.desired_data_point = seconds * SAMPLE_RATE
        self.table = self.openTable()
        self.overlap_dot = overlap_dot    

    def makeData(self):
        
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]
        
        self.startMakingData(afTotal, 2, 3)
        self.startMakingData(noiseTotal, 0, 1)
        self.startMakingData(otherTotal, 4, 5)
        self.startMakingData(normalTotal, 6, 7)
        newData = np.array(self.newData)
        newLabel = np.array(self.newLabel)
        return newData, newLabel

    def startMakingData(self, totalDataInThisClass, dataIndex, labelIndex):
        
        self.CLASS_AMOUNT = 0
        
        for i in range(totalDataInThisClass):
            dataLen = len(self.openData(self.table.iloc[i,dataIndex]))
            self.dataRemainder = 0
            if(dataLen >= self.desired_data_point):
                self.reduceData(i, dataIndex, labelIndex)
            if(self.dataRemainder != 0 ):
                self.makeInsuffitientData(i, dataIndex, labelIndex)
            if(dataLen < self.desired_data_point):
                self.increaseData(i, dataIndex, labelIndex)
        
        print(str(self.table.iloc[i,labelIndex]) + " is " + str(self.CLASS_AMOUNT))

    def reduceData(self,i ,dataIndex, labelIndex):
        j = 1
        self.previous_j = 0
        data = self.openData(self.table.iloc[i,dataIndex])
        label = self.table.iloc[i,labelIndex]
        while j <= len(data):
            self.dataRemainder += 1
            if self.dataRemainder == self.desired_data_point:
                self.CLASS_AMOUNT += 1
                self.newData.append(data[self.previous_j : j])
                j = j - self.overlap_dot
                self.previous_j = j
                self.newLabel.append(self.ONE_HOT_ENCODE_LABEL[label])
                self.dataRemainder = 0
            j += 1

    def makeInsuffitientData(self, i, dataIndex, labelIndex):
        data = self.openData(self.table.iloc[i,dataIndex])
        label = self.table.iloc[i,labelIndex]
        self.newData.append(data[len(data)-self.desired_data_point  : ])
        self.newLabel.append(self.ONE_HOT_ENCODE_LABEL[label])
        self.CLASS_AMOUNT += 1

    def increaseData(self, i, dataIndex, labelIndex):
        data = self.openData(self.table.iloc[i,dataIndex])
        label = self.table.iloc[i,labelIndex]
        tmp = data
        numOfBatch = self.desired_data_point // len(data)
        leftData = self.desired_data_point % len(data)
        while numOfBatch > 1 :
            tmp = np.append(tmp, data)
            numOfBatch -= 1
        if(leftData != 0):
            tmp = np.append(tmp, data[ : leftData ])
        self.newData.append(tmp)
        self.newLabel.append(self.ONE_HOT_ENCODE_LABEL[label])
        self.CLASS_AMOUNT += 1
        
    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV

    def openData(self, filename):
        index = wf.rdsamp(ECG_folder_path + filename)
        record = index[0]
        record.shape = (record.shape[0], )
        return record

# print("30s : ")
# newData, newLabel = generateData(30).makeData()
# print(newLabel.shape)
# print("9s : ")
# newData, newLabel = generateData(9, 300).makeData()
# print("9s overlap_dot: ")
# newData, newLabel = generateData(9, 300, 1350).makeData()
        
