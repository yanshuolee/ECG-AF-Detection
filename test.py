import numpy as np
import wfdb as wf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
np.set_printoptions(suppress=True)


import data_datacut as dp
newData, newLabel = dp.datacut(9,300,10).newdata()
# print(newData.shape)
# print(newLabel.shape)

trainData = newData.reshape(newData.shape[0], newData.shape[1] ,1)
## to check:
# 1. train shape
print(trainData.shape)

model = Sequential() 
model.add(Conv1D(filters = 512, kernel_size = 3, input_shape = trainData[1:3], activation = "relu"))
model.add(MaxPooling1D(pool_size = 100))
model.add(Dense(4, activation = "softmax"))
print(model.summary())

model.compile(optimizer = "adam", loss = "categorical_crossentropy", matrics = ["accuracy"])
model.fit(x = newData, y = newLabel, validation_split=0.2, epochs=10, batch_size=100, verbose=2)




