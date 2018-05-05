import numpy as np
import wfdb as wf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import np_utils
np.set_printoptions(suppress=True)


import data_processing as dp
newData, newLabel = dp.makeData(30)

trainData = newData.reshape((newData.shape[0], newData.shape[1], 1))
trainLabel = np_utils.to_categorical(newLabel, 4)

print('Train data shape:' , trainData.shape)
print('Train label shape:', trainLabel.shape)


model = Sequential() 
model.add(Conv1D(filters = 512, kernel_size = 3, input_shape = (newData.shape[1], 1), activation = "relu"))
model.add(MaxPooling1D(pool_size = 100))
model.add(Flatten())
model.add(Dense(4, activation = "softmax"))
print(model.summary())

model.compile(optimizer ='adam', loss = "categorical_crossentropy", metrics=['accuracy'])
model.fit(x = trainData, y = trainLabel, epochs=10, batch_size=100, verbose=2)




