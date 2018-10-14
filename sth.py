import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Activation, Input
from keras.utils import np_utils
import tensorflow as tf
np.set_printoptions(suppress=True)

data = np.load("/home/hsiehch/30s/2D_spect_img/data_RGB.npy")
labels = np.load('/home/hsiehch/30s/2D_spect_img/labels.npy')

data = data / 255
labels = np_utils.to_categorical(labels, 4)

print(data.shape)
print(labels.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels)

img_input = Input(shape=(data.shape[1], data.shape[2], 3))
# Block 1
x = Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1')(x)
x = Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1')(x)
x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2')(x)
x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv1')(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv2')(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',)(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',)(x)
x = Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',)(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)


x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(256, activation='relu', name='fc3')(x)
x = Dense(4, activation='softmax', name='predictions')(x)
model = Model(img_input, x, name='vgg')
print(model.summary())

model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=['accuracy'])
batch_s = 80
train_history = model.fit(x = X_train, 
                          y = y_train,
                          epochs=100,
                          validation_data=(X_test, y_test),
                          batch_size=batch_s*1, 
                          verbose=1)

print('Finish training!')

