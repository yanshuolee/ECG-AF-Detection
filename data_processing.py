'''
created by Y.S.L
'''
import numpy as np
import wfdb as wf
import pandas as pd
import pylab as pl
import json
np.set_printoptions(suppress=True) #prevent numpy exponential notation on print, default False


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
    THRESH_1 = 9 * SAMPLE_RATE
    ONE_HOT_ENCODE_LABEL = {'A':'0,0,0,1', '~':'0,0,1,0', 'N':'0,1,0,0', 'O':'1,0,0,0'}
    global txtFile
    global txtFile_count_data
    global organized_train_file
    global organized_test_file
    
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

        # print(dataInfoSet)
        lists = sorted(dataInfoSet.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists)
        # q = sum(y)
        # print('Total data: ' + chr(q))
        pl.ylim(0,10)
        pl.plot(x, y)
        pl.show()


    def calculateData(self, Total, dataIndex):
        
        for i in range(Total):
            data = self.openData(self.table.iloc[i,dataIndex])
            dataLen = len(data)
            dataDuration = dataLen / 300
            dataDuration = round(dataDuration, 2)
            if dataDuration not in dataInfoSet:
                dataInfoSet[dataDuration] = 1
            else:
                dataInfoSet[dataDuration] += 1
            
    def modifyDataTo9s(self):
        TXT_PATH = 'data_in_9s.txt'
        self.txtFile = open(TXT_PATH, 'w+')
        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]

        # self.matchData9s_ver(2, 2, 3) # for testing

        self.matchData9s_ver(afTotal, 2, 3)
        self.matchData9s_ver(noiseTotal, 0, 1)
        self.matchData9s_ver(otherTotal, 4, 5)
        self.matchData9s_ver(normalTotal, 6, 7)

        self.txtFile.close()




    def matchData9s_ver(self, Total, dataIndex, labelIndex):
        recursive = 0
        for i in range(Total):
            previous_j = 0
            j = 1
            count = 0
            newArr = np.array([])
            data = self.openData(self.table.iloc[i,dataIndex])
            label = self.table.iloc[i,labelIndex]

            while j <= len(data):
                count = count + 1
                if count == self.THRESH_1:
                    newArr = np.append(newArr, data[previous_j : j])
                    j = j - recursive
                    previous_j = j
                    self.writeToFile(newArr, label)
                    count = 0
                j = j + 1
            if(count != 0 & len(data)-previous_j >= self.THRESH_1/2):
                newArr = np.append(newArr, data[len(data)-self.THRESH_1  : ])
                self.writeToFile(newArr, label)
    
    
    
    
    def modifyDataTo30s(self):
        
        TXT_PATH = 'data_in_30s.txt'
        self.txtFile = open(TXT_PATH, 'w+')

        TXT_PATH = 'data_in_30s_info.txt'
        self.txtFile_count_data = open(TXT_PATH, 'w+')


        afTotal = self.table.count(axis = 0)[3]
        noiseTotal = self.table.count(axis = 0)[1]
        otherTotal = self.table.count(axis = 0)[5]
        normalTotal = self.table.count(axis = 0)[7]

        self.matchData(afTotal, 2, 3)
        self.matchData(noiseTotal, 0, 1)
        self.matchData(otherTotal, 4, 5)
        self.matchData(normalTotal, 6, 7)

        self.txtFile.close()
        self.txtFile_count_data.close()


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
        
        print(str(count) + ' ' + label + ' in total.')
        # tmp = [label, count]
        # json.dump(tmp, self.txtFile_count_data)
        self.txtFile_count_data.write(str(count) + ',' + label + '\n')
        
                      

    def writeToFile(self, Arr, lab):
        # print('arr length:'+str(len(Arr)))
        # print(Arr)
        label = self.ONE_HOT_ENCODE_LABEL[lab]
        arrToStr = ','.join(str(e) for e in Arr)
        arrToStr = arrToStr + ',' + label
        self.txtFile.write(arrToStr)
        self.txtFile.write('\n')
        

    def makeData(self,data_path, data_info_path):
        numOfAf, numOfNormal, numOfOther, numOfNoise = self.numOfDataForTraining(data_info_path)
        self.startMakingData(numOfAf, numOfNormal, numOfOther, numOfNoise, data_path, data_info_path)
        print('Finish making dataset. Check the file!')

    def numOfDataForTraining(self, path):
        
        data_total_txt = open(path, 'r')
        txt = data_total_txt.readlines()
        output_arr = np.array([i.split(',') for i in txt])
        
        af = int(int(output_arr[0][0]) * self.percentageForTrainingData)
        noise = int(int(output_arr[1][0]) * self.percentageForTrainingData)
        other = int(int(output_arr[2][0]) * self.percentageForTrainingData)
        normal = int(int(output_arr[3][0]) * self.percentageForTrainingData)

        return af, normal, other, noise

    def startMakingData(self, numOfAf, numOfNormal, numOfOther, numOfNoise, path, data_info_path):
        
        data = self.getData(path)
        afTotal, noiseTotal, otherTotal, normalTotal = self.getDataLen(data_info_path)
        self.organized_train_file = open('train_data.txt', 'w+')
        self.organized_test_file = open('test_data.txt', 'w+')
        
        start_point = 0
        self.appendData(data, afTotal, numOfAf, start_point)
        start_point += afTotal
        self.appendData(data, noiseTotal, numOfNoise, start_point)
        start_point += noiseTotal
        self.appendData(data, otherTotal, numOfOther, start_point)
        start_point += otherTotal
        self.appendData(data, normalTotal, numOfNormal, start_point)

        self.organized_train_file.close()
        self.organized_test_file.close()


    def getData(self, path):

        openData = open(path, 'r')
        readData = openData.readlines()
        data = np.array([i.split(',') for i in readData])
        
        return data

    def getDataLen(self, path):
        openData = open(path, 'r')
        readData = openData.readlines()
        data = np.array([i.split(',') for i in readData])
        af = data[0][0]
        noise = data[1][0]
        other = data[2][0]
        normal = data[3][0]
        return af, noise, other, normal

    def appendData(self, data, Total, trainingPart, start_point):
        
        for i in range(Total):
            if i < trainingPart:
                self.organized_train_file.write(data[i]+'\n')
            else:
                self.organized_test_file.write(data[i]+'\n')

    def openTable(self):
        dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
        return dataFromCSV

    def openData(self, filename):
        index = wf.rdsamp(ECG_folder_path + filename)
        record = index[0]
        record.shape = (record.shape[0],)
        
        return record
