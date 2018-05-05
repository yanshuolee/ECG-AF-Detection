import numpy as np
import wfdb as wf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import np_utils
np.set_printoptions(suppress=True)


import data_processing as dp
newData, newLabel = dp.generateData(30).makeData()
print(newLabel.shape)


trainData = newData.reshape((newData.shape[0], newData.shape[1], 1))

tmp = newLabel.reshape((newLabel.shape[0], 1))
print(tmp.shape)
trainLabel = np_utils.to_categorical(tmp, 4)

# print(trainData)
# print(trainLabel)
print(trainData.shape)
print(trainLabel.shape)


model = Sequential() 
model.add(Conv1D(filters = 512, kernel_size = 3, input_shape = (newData.shape[1], 1), activation = "relu"))
model.add(MaxPooling1D(pool_size = 100))
model.add(Dense(4, activation = "softmax"))
print(model.summary())

model.compile(optimizer = "adam", loss = "categorical_crossentropy", matrics = ["accuracy"])
model.fit(x = trainData, y = trainLabel, epochs=10, batch_size=100, verbose=2)




