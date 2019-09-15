import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, Dropout
from keras.layers import Activation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import KFold
np.set_printoptions(suppress=True)

def history_display(path, hist, train, validation):
    plt.figure()
    plt.plot(hist.history[train])
    plt.plot(hist.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path+"/"+train+'.png')
    
def show_plot(path, hist):
    history_display(path, hist, 'acc', 'val_acc')
    history_display(path, hist, 'loss', 'val_loss')

def write_file(path, evaluation, f1_scores, model_structure):
    with open(path, "w+") as writer:
        writer.write('Loss: {:.3f}'.format(evaluation[0])+"\n")
        writer.write('Accuracy: {:.3f}'.format(evaluation[1])+"\n")
        writer.write('F1[A]: '+str(f1_scores[0])+"\n")
        writer.write('F1[~]: '+str(f1_scores[1])+"\n")
        writer.write('F1[N]: '+str(f1_scores[2])+"\n")
        writer.write('F1[O]: '+str(f1_scores[3])+"\n")
        model_structure.summary(print_fn=lambda x: writer.write(x + '\n'))
        
def get_data():
    trainD = np.load("/home/hsiehch/30s/train_data.npy")
    trainL = np.load("/home/hsiehch/30s/train_label.npy")
    validationD = np.load("/home/hsiehch/30s/validation_data.npy")
    validationL = np.load("/home/hsiehch/30s/validation_label.npy")
    testD = np.load("/home/hsiehch/30s/test_data.npy")
    testL = np.load("/home/hsiehch/30s/test_label.npy")

    trainD = np.append(trainD, validationD, axis=0)
    trainL = np.append(trainL, validationL, axis=0)
    trainD = np.append(trainD, testD, axis=0)
    trainL = np.append(trainL, testL, axis=0)

    trainData = trainD.reshape((trainD.shape[0], trainD.shape[1], 1))
    trainLabel = np_utils.to_categorical(trainL, 4)
    print('Train Data:', trainData.shape)
    print('Train Label: ', trainLabel.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    print(kf)

    training_data = []
    training_label = []
    validation_cate_label = []
    validation_data = []
    validation_label = []

    for train_index, test_index in kf.split(trainData):
        print('trian:', train_index, 'len', len(train_index), 'test:', test_index, 'len', len(test_index))
        training_data.append(trainData[train_index])
        training_label.append(trainLabel[train_index])
        validation_data.append(trainData[test_index])
        validation_label.append(trainLabel[test_index])
        validation_cate_label.append(trainL[test_index])

    training_data = np.array(training_data)
    training_label = np.array(training_label)
    validation_data = np.array(validation_data)
    validation_label = np.array(validation_label)
    validation_cate_label = np.array(validation_cate_label)
    
    return training_data, training_label, validation_data, validation_label, validation_cate_label


def create_model(learning_rate, bs, ks, num_layer):
    num_filter = 32
    
    model = Sequential()
    # baseline
    model.add(Conv1D(filters = num_filter, kernel_size = ks, input_shape = (9000, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2))
    
    for i in range(2,num_layer+1):
        try:
            if i==num_layer:
                model.add(Conv1D(filters = num_filter, kernel_size = ks))
                model.add(Activation('relu'))
                break
            if i%2 != 0:
                num_filter = num_filter *2
            model.add(Conv1D(filters = num_filter, kernel_size = ks))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size = 2))
            if i in [6, 8, 9]:
                model.add(Dropout(0.5))
        except ValueError:
            print("model overflow[lr, bs, ks, #layer]: ",[learning_rate, bs, ks, i+1])
            return False
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(4, activation = "softmax"))
    
    adam = Adam(lr = learning_rate)
    model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics=['accuracy'])
    
    return model