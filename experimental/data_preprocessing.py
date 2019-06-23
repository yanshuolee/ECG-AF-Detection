import numpy as np
import wfdb as wf
import pandas as pd

table_path = '/home/hsiehch/ECG-project/table.csv'
ECG_folder_path = '/home/hsiehch/dataset/'

class makeData():
    
    newData = []
    newLabel = []
    ONE_HOT_ENCODE_LABEL = {'A':0, '~':1, 'N':2, 'O':3}
    LABEL_TOTAL_COUNT = []
    

    def __init__(self, seconds, percentageForTraining, percentageForValidation, percentageForTesting, overlap_dot = 0):
        SAMPLE_RATE = 300
        self.desired_data_point = seconds * SAMPLE_RATE
        self.table = self.openTable()
        self.overlap_dot = overlap_dot
        self.percentageForTraining = percentageForTraining
        self.percentageForValidation = percentageForValidation
        self.percentageForTesting = percentageForTesting
        self.dataPosition = []
        if(self.percentageForTraining+self.percentageForValidation+self.percentageForTesting != 1):
            raise ValueError ("Wrong Proportion!")
        

    def main(self):
        
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
        
        T_d, T_l, V_d, V_l, Te_d, Te_l = self.splitData()
        print(T_d.shape)
        print(T_l.shape)
        print(V_d.shape)
        print(V_l.shape)
        print(Te_d.shape)
        print(Te_l.shape)

        return T_d, T_l, V_d, V_l, Te_d, Te_l

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
        
        self.LABEL_TOTAL_COUNT.append(self.CLASS_AMOUNT)
        print(str(self.table.iloc[i,labelIndex]) + ": " + str(self.CLASS_AMOUNT))

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
        
    def splitData(self):
        train_data = []
        train_label = []
        val_data = []
        val_label = []
        test_data = []
        test_label = []
        
        dataIndex = self.getDataPosition()
        for i in dataIndex:
            for j in range(i[0], i[3]):
                if j < i[1]:
                    train_data.append(self.newData[j])
                    train_label.append(self.newLabel[j])
                elif j >= i[1] and j < i[2]:
                    val_data.append(self.newData[j])
                    val_label.append(self.newLabel[j])
                else:
                    test_data.append(self.newData[j])
                    test_label.append(self.newLabel[j])

        train_data = np.array(train_data)
        train_label = np.array(train_label)
        val_data = np.array(val_data)
        val_label = np.array(val_label)
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        return train_data, train_label, val_data, val_label, test_data, test_label

    def getDataPosition(self):
        amount_for_training = [int(i*self.percentageForTraining)+1 for i in self.LABEL_TOTAL_COUNT]
        amount_for_val = [int(i*self.percentageForValidation)+1 for i in self.LABEL_TOTAL_COUNT]
        startPoint = 0
        
        for i in range(len(self.LABEL_TOTAL_COUNT)):
            trainPoint = amount_for_training[i]+startPoint
            validationPoint = amount_for_training[i]+amount_for_val[i]+startPoint
            endPoint = self.LABEL_TOTAL_COUNT[i]+startPoint
            self.dataPosition.append([startPoint, trainPoint, validationPoint, endPoint])
            startPoint += self.LABEL_TOTAL_COUNT[i]
        
        print('split train data index: ', self.dataPosition)
        return self.dataPosition

    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV

    def openData(self, filename):
        index = wf.rdsamp(ECG_folder_path + filename)
        record = index[0]
        record.shape = (record.shape[0], )
        return record
        
