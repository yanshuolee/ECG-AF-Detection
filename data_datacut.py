import numpy as np
import wfdb as wf
import pandas as pd
from numba import jit
from timeit import default_timer as timer

table_path = 'table.csv'
ECG_folder_path = '/home/hsiehch/dataset/'

class datacut():
    newData = []
    newLabel = []
    def __init__(self, sampling, frequency,recursive=0):
        self.datatake = sampling*frequency
        self.table = self.openTable()
        self.recursive = recursive    
    def newdata(self):
        
        start = timer()
        
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]
        self.datamodify(afTotal, 2, 3)
        self.datamodify(noiseTotal, 0, 1)
        self.datamodify(otherTotal, 4, 5)
        self.datamodify(normalTotal, 6, 7)
        newData = np.array(self.newData)
        newLabel = np.array(self.newLabel)
        
        vector_add_cpu_time = timer() - start
        print("GPU function took %f seconds." % vector_add_cpu_time)
        
        return newData, newLabel
        
        
    
    @jit
    def datamodify(self, Total, dataIndex, labelIndex):
        for i in range(Total):
            previous_j = 0
            j = 1
            count = 0
            data = self.openData(self.table.iloc[i,dataIndex])
            while j <= len(data):
                count = count + 1
                if count == self.datatake:
                    self.newData.append(data[previous_j : j])
                    j = j - self.recursive
                    previous_j = j
                    self.newLabel.append([self.table.iloc[i,labelIndex]])
                    count = 0
                j = j + 1
            if(count != 0 & len(data)-previous_j >= self.datatake/2):
                self.newData.append(data[len(data)-self.datatake  : ])
                self.newLabel.append([self.table.iloc[i,labelIndex]])
    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV
    def openData(self, filename):
        index = wf.rdsamp(ECG_folder_path + filename)
        record = index[0]
        record.shape = (record.shape[0],)
        return record

# newData, newLabel = datacut(9,300,10).newdata()
# print(len(newData[6]))
# print(newData)
# # print(newLabel)
        
