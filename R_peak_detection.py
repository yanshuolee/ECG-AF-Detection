import numpy as np
import wfdb as wf
import pandas as pd
from biosppy.signals import ecg
from toolkit import splitData
from toolkit import openFile
from numba import jit
ECG_folder_path = '/home/hsiehch/dataset/'
table_path = 'labels.csv'

class RpeakData():
    
    newData = []
    newLabel = []
    ONE_HOT_ENCODE_LABEL = {'A':0, '~':1, 'N':2, 'O':3}
    LABEL_TOTAL_COUNT = []
    MAX = 0
    
    def __init__(self):
        self.table = self.openTable()
    
    def checkMax(self, R_Peak_Array):
        
        for i in range(1, len(R_Peak_Array)):
            length = R_Peak_Array[i] - R_Peak_Array[i-1]
            if length > self.MAX:
                self.MAX = length
    @jit(parallel=True)            
    def getRPosition(self):
        data_total = self.table.count(axis = 0)[0]
        
        for i in range(data_total):
            data = openFile.openData(ECG_folder_path, self.table.iloc[i,0])
            peaks = ecg.christov_segmenter(data, 300)
            print(peaks)
            
            self.checkMax(peaks)
            tmp = []
            
            for index in range(1, len(peaks)):
                tmp.append([data[peaks[index-1] : peaks[index]+1]])
                
            self.newData.append(tmp)
            self.newLabel.append([self.ONE_HOT_ENCODE_LABEL[self.table.iloc[i,1]]])
        
        print(self.MAX)
    
        for i in range(self.newData):
            
            datum = []
            for wave in self.newData[i]:
                tmp = np.pad(wave, (0, self.MAX - len(wave)), 'constant')
                datum = datum+tmp
            
            self.newData[i] = datum
            
            print(self.newData[i])
            
        return self.newData, self.newLabel
        
 
    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV