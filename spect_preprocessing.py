import numpy as np
import pylab as plt
from numba import jit

@jit
def to_Spectrogram():
    
    trainData = []
    validationData = []
#     testData = []

    trainD = np.load("/home/hsiehch/9s/train_data.npy")
    validationD = np.load("/home/hsiehch/9s/validation_data.npy")
#     testD = np.load("/home/hsiehch/9s/test_data.npy")
    length = 75
    overlap = 37
    
    print(trainD.shape)
    w = 0
    for data in trainD:
        w += 1
        if w%1000 == 0:
            print(w)
        i = plt.specgram(data, Fs =300, NFFT=length, noverlap=overlap, scale_by_freq = True, sides = 'default')
        trainData.append(i[0])
    
    w = 0
    print(validationD.shape)
    for data in validationD:
        w += 1
        if w%1000 == 0:
            print(w)
        i = plt.specgram(data,Fs =300, NFFT=length, noverlap=overlap, scale_by_freq = True, sides = 'default')
        validationData.append(i[0])
    
#     w = 0
#     print(testD.shape)
#     for data in testD:
#         w += 1
#         print(w)
#         i = plt.specgram(data,Fs =300, NFFT=length, noverlap=overlap, scale_by_freq = True, sides = 'default')
#         testData.append(i[0])
    
    trainData = np.asarray(trainData)
    validationData = np.asarray(validationData)
#     testData = np.asarray(testData)

    return trainData, validationData
