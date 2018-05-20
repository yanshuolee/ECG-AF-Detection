from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D
import variable_length as vl
import numpy as np
trainD, trainL, validateD, validateL, testD, testL = vl.main(0.5, 0.2, 0.3)


model = Sequential()
model.add(Conv1D(filters = 128, kernel_size = 7, input_shape = (None, 1), activation = "relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(4, activation = "softmax"))
print(model.summary())

def makeArray(shape):
    # print(trainD[shape][2])
    print(len(trainD[shape]))
    tmpD = []
    tmpL = []
    for data in trainD[shape]:
        tmpD.append(data)
    for label in trainL[shape]:
        tmpL.append(label)
    dataArr = np.array(tmpD)
    labelArr = np.array(tmpL)
    print(labelArr.shape)
    return dataArr, labelArr

model.compile(optimizer ='adam', loss = "categorical_crossentropy", metrics=['accuracy'])
for i in trainD:
    t_D, t_L = makeArray(i)
    model.fit(x = t_D, 
            y = t_L,
            epochs=2,
            batch_size=16,
            verbose=2)
    print('Finish shape!')
