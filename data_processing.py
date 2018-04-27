'''
created by Y.S.L
'''
import numpy as np
import wfdb as wf
import pandas as pd
import pylab as pl

table_path = 'table.csv'
ECG_folder_path = 'dataset/'

class generateData():
    '''available method: modifyData() / dataInfo() / '''
    trainingData = []
    trainingLabel = []
    testData = []
    testLabel = []

    SAMPLE_RATE = 300
    THRESH = 30 * SAMPLE_RATE
    ONE_HOT_ENCODE_LABEL = {'A':'0,0,0,1', '~':'0,0,1,0', 'N':'0,1,0,0', 'O':'1,0,0,0'}
    
    
    def __init__(self, percentageForTrainingData):
        
        if percentageForTrainingData < 1 and percentageForTrainingData > 0:
            self.percentageForTrainingData = percentageForTrainingData
            self.table = self.openTable()
        else:
            raise ValueError ("the range should be 0 to 1!")


    def dataInfo(self):
        
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]
        global dataInfoSet
        dataInfoSet = {}

        self.calculateData(afTotal, 2)
        self.calculateData(noiseTotal, 0)
        self.calculateData(otherTotal, 4)
        self.calculateData(normalTotal, 6)

        lists = sorted(dataInfoSet.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists)
        q = sum(y)
        print('Total data: ' + chr(q))

        pl.plot(x, y)
        pl.show()


    def calculateData(self, Total, dataIndex):
        
        for i in range(Total):
            data = self.openData(self.table.iloc[i,dataIndex])
            dataLen = len(data)
            dataDuration = dataLen / self.THRESH
            if dataDuration not in dataInfoSet:
                dataInfoSet[dataDuration] = 1
            else:
                dataInfoSet[dataDuration] += 1
            

    def modifyData(self):
        
        TXT_PATH = 'data_in_30s.txt'
        global txtFile
        txtFile = open(TXT_PATH, 'w+')
        # afTotal = self.table.count(axis = 0)[3]
        # noiseTotal = self.table.count(axis = 0)[1]
        # otherTotal = self.table.count(axis = 0)[5]
        # normalTotal = self.table.count(axis = 0)[7]

        self.matchData(2, 2, 3)
        self.matchData(2, 0, 1)
        self.matchData(2, 4, 5)
        self.matchData(2, 6, 7)

        txtFile.close()


    def matchData(self, Total, dataIndex, labelIndex):
        count = 0
        for i in range(Total):
            
            data = self.openData(self.table.iloc[i,dataIndex])
            label = self.table.iloc[i,labelIndex]
            # print('label:'+str(label))
            dataLen = len(data)
            # print('data length:'+ str(dataLen))
            newArr = np.array([])

            if dataLen > self.THRESH:
                LOOPS = dataLen // self.THRESH
                if dataLen % self.THRESH != 0:
                    LOOPS += 1
                for j in range(LOOPS):
                    if j == LOOPS:
                        newArr = np.append(newArr, data[dataLen-self.THRESH:])
                        self.writeToFile(newArr, label)
                        count += 1
                    else:
                        newArr = data[j*self.THRESH: j*self.THRESH+self.THRESH]
                        self.writeToFile(newArr, label)
                        count += 1
            else:
                LOOPS = self.THRESH // dataLen
                if self.THRESH % dataLen == 0:
                    for j in range(LOOPS):
                        newArr = np.append(newArr, data[j*dataLen: j*dataLen+dataLen])
                        self.writeToFile(newArr, label)
                        count += 1
                else:
                    for j in range(LOOPS+1):
                        if j == LOOPS:
                            newArr = np.append(newArr, data[dataLen-self.THRESH:])
                            self.writeToFile(newArr, label)
                            count += 1
                        else:
                            newArr = np.append(newArr, data[j*dataLen: j*dataLen+dataLen])
                            self.writeToFile(newArr, label)
                            count += 1
        
        print(str(count) + label + ' in total.')
                      

    def writeToFile(self, Arr, lab):
        # print('arr length:'+str(len(Arr)))
        # print(Arr)
        label = self.ONE_HOT_ENCODE_LABEL[lab]
        arrToStr = ','.join(str(e) for e in Arr)
        arrToStr = arrToStr + ',' + label
        txtFile.write(arrToStr)
        txtFile.write('\n')
        

    def makeData(self):
        numOfAf, numOfNormal, numOfOther, numOfNoise = self.numOfDataForTraining()
        self.startMakingData(numOfAf, numOfNormal, numOfOther, numOfNoise)
        trainingData = np.array(self.trainingData)
        trainingLabel = np.array(self.trainingLabel)
        testData = np.array(self.testData)
        testLabel = np.array(self.testLabel)
        return trainingData, trainingLabel, testData, testLabel

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
                self.trainingData.append(data)
                self.trainingLabel.append([self.table.iloc[i,labelIndex]])

            else:
                data = self.openData(self.table.iloc[i,dataIndex])
                self.testData.append(data)
                self.testLabel.append([self.table.iloc[i,labelIndex]])

    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV

    def openData(self, filename):
        index = wf.rdsamp(ECG_folder_path + filename)
        record = index[0]
        record.shape = (record.shape[0],)
        
        return record
