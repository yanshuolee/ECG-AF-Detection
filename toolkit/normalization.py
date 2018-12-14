import numpy as np
from sklearn.preprocessing import normalize

def normalize_arr(input_arr, trace = False):
    dataIndex = []
    dataList = []
    for i in range(len(input_arr)):
        for j in range(1, len(input_arr[i])):
            if abs(input_arr[i][j-1] - input_arr[i][j]) > 1:
                dataIndex.append(i)
                dataList.append(input_arr[i])
                break
    
    if trace:
        print('Problematic data index: ', dataIndex)
    
    dataList = normalize(dataList)
    
    for i, j in zip(dataIndex, dataList):
        input_arr[i] = j
    
    print('Normalization done!')
    return input_arr