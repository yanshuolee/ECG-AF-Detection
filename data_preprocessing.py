import numpy as np
import pandas as pd
from toolkit import processData
from toolkit import splitData
import importlib

table_path = 'table.csv'


class makeData():
        
    LABEL_TOTAL_COUNT = []
    
    def __init__(self, seconds, percentageForTraining, percentageForValidation, percentageForTesting, overlap_dot = 0):
        SAMPLE_RATE = 300
        self.desired_data_point = seconds * SAMPLE_RATE
        self.table = self.openTable()
        self.overlap_dot = overlap_dot
        self.percentageForTraining = percentageForTraining
        self.percentageForValidation = percentageForValidation
        self.percentageForTesting = percentageForTesting
        if(self.percentageForTraining+self.percentageForValidation+self.percentageForTesting != 1):
            raise ValueError ("Wrong Proportion!")
        
    def main(self):
        
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]
        
        CLASS_AMOUNT_AF, newData, newLabel = processData.makeData(afTotal, 2, 3, self.table, self.desired_data_point, self.overlap_dot)
        importlib.reload(processData)
        
        CLASS_AMOUNT_Noise, data, label = processData.makeData(noiseTotal, 0, 1, self.table, self.desired_data_point, self.overlap_dot)
        newData = np.append(newData, data, axis=0)
        newLabel = np.append(newLabel, label, axis=0)
        importlib.reload(processData)
        
        CLASS_AMOUNT_Other, data, label = processData.makeData(otherTotal, 4, 5, self.table, self.desired_data_point, self.overlap_dot)
        newData = np.append(newData, data, axis=0)
        newLabel = np.append(newLabel, label, axis=0)
        importlib.reload(processData)
        
        CLASS_AMOUNT_Normal, data, label = processData.makeData(normalTotal, 6, 7, self.table, self.desired_data_point, self.overlap_dot)
        newData = np.append(newData, data, axis=0)
        newLabel = np.append(newLabel, label, axis=0)

        LABEL_TOTAL_COUNT = [CLASS_AMOUNT_AF, CLASS_AMOUNT_Noise, CLASS_AMOUNT_Other, CLASS_AMOUNT_Normal]      
   
        T_d, T_l, V_d, V_l, Te_d, Te_l = splitData(newData, newLabel, LABEL_TOTAL_COUNT, self.percentageForTraining, self.percentageForValidation)
        print(T_d.shape)
        print(T_l.shape)
        print(V_d.shape)
        print(V_l.shape)
        print(Te_d.shape)
        print(Te_l.shape)

        return T_d, T_l, V_d, V_l, Te_d, Te_l
 
    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV

