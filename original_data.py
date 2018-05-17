'''
created by Y.S.L
'''
import numpy as np
import wfdb as wf
import pandas as pd

table_path = 'table.csv'
ECG_folder_path = '/home/hsiehch/dataset/'

class makeData():
    
    trainingData = []
    traingLabel = []
    testData = []
    testLabel = []
    ONE_HOT_ENCODE_LABEL = {'A':0, '~':1, 'N':2, 'O':3}
    

    def __init__(self, percentageForTrainingData):
        if percentageForTrainingData < 1 and percentageForTrainingData > 0:
            self.percentageForTrainingData = percentageForTrainingData
            self.table = self.openTable()
#             self.generateData()
        else:
            raise ValueError ("the range should be 0 to 1!")

    def makeVariableSizeData(self):
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]
        
        dataList = []
        for i in range(afTotal):
            dataList.append(self.openData(self.table.iloc[i,2]))
        print(dataList)
    
    def generateData(self):
        numOfAf, numOfNormal, numOfOther, numOfNoise = self.numOfDataForTraining()
        self.startMakingData(numOfAf, numOfNormal, numOfOther, numOfNoise)
        
        t_D = np.array(self.trainingData)
        t_L = np.array(self.trainingData)
        test_D = np.array(self.trainingData)
        test_L = np.array(self.trainingData)
        return t_D, t_L, test_D, test_L

    def numOfDataForTraining(self):
        
        af = int(self.table.count(axis = 0)[3] * self.percentageForTrainingData)
        noise = int(self.table.count(axis = 0)[1] * self.percentageForTrainingData)
        other = int(self.table.count(axis = 0)[5] * self.percentageForTrainingData)
        normal = int(self.table.count(axis = 0)[7] * self.percentageForTrainingData)

        return af, normal, other, noise

    def startMakingData(self, numOfAf, numOfNormal, numOfOther, numOfNoise):
        
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]
        
        self.appendData(afTotal, 2, 3, numOfAf)
        self.appendData(noiseTotal, 0, 1, numOfNoise)
        self.appendData(otherTotal, 4, 5, numOfOther)
        self.appendData(normalTotal, 6, 7, numOfNormal)

    def appendData(self, Total, dataIndex, labelIndex, trainingPart):
        
        for i in range(Total):
            if i < trainingPart:
                data = self.openData(self.table.iloc[i,dataIndex])
                label = self.table.iloc[i,labelIndex]
                self.trainingData.append(data)
                self.traingLabel.append(self.ONE_HOT_ENCODE_LABEL[label])

            else:
                data = self.openData(self.table.iloc[i,dataIndex])
                self.table.iloc[i,labelIndex]
                self.testData.append(data)
                self.testLabel.append(self.ONE_HOT_ENCODE_LABEL[label])

    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV

    def openData(self, filename):
        index = wf.rdsamp(ECG_folder_path + filename)
        record = index[0]
        
        return record
