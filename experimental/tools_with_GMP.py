# TODO fix agg problem using https://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
# import matplotlib
# matplotlib.use('Agg')
import wfdb as wf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, GlobalMaxPooling1D, Dropout, MaxPooling1D
from keras.layers import Activation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
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
    table_path = '/home/hsiehch/ECG-project/table.csv'
    ECG_folder_path = '/home/hsiehch/dataset/'
    ONE_HOT_ENCODE_LABEL = {'A':0, '~':1, 'N':2, 'O':3}
    dataFromCSV = pd.read_csv(table_path,dtype='str',header=None)
    
    afTotal = dataFromCSV.count(axis = 0)[3]
    noiseTotal = dataFromCSV.count(axis = 0)[1]
    otherTotal = dataFromCSV.count(axis = 0)[5]
    normalTotal = dataFromCSV.count(axis = 0)[7]
    global data
    global label
    data = []
    label = []
    def generate(totalDataInThisClass, dataIndex, labelIndex):
        global data
        global label
        for i in range(totalDataInThisClass):
            index = wf.rdsamp(ECG_folder_path + dataFromCSV.iloc[i,dataIndex])
            record = index[0]
            record.shape = (record.shape[0], 1)
            data.append(record)
            
            label_index = dataFromCSV.iloc[i,labelIndex]
            label.append(ONE_HOT_ENCODE_LABEL[label_index])
            
    generate(afTotal, 2, 3)
    generate(noiseTotal, 0, 1)
    generate(otherTotal, 4, 5)
    generate(normalTotal, 6, 7)
    
    data = np.array(data)
    label = np.array(label)
    _label = np_utils.to_categorical(label, 4)
    
    print("data: {}".format(len(data)))
    print("label: {}".format(len(_label)))
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    print(kf)
    
    training_data = []
    training_label = []
    validation_cate_label = []
    validation_data = []
    validation_label = []

    for train_index, test_index in kf.split(data, label):
        print('trian:', train_index, 'len', len(train_index), 'test:', test_index, 'len', len(test_index))
        training_data.append(data[train_index])
        training_label.append(_label[train_index])
        validation_data.append(data[test_index])
        validation_label.append(_label[test_index])
        validation_cate_label.append(label[test_index])

    training_data = np.array(training_data)
    training_label = np.array(training_label)
    validation_data = np.array(validation_data)
    validation_label = np.array(validation_label)
    validation_cate_label = np.array(validation_cate_label)
    
    return training_data, training_label, validation_data, validation_label, validation_cate_label

def create_model(learning_rate, bs, ks, num_layer):
    num_filter = 32
    
    in_data = tf.keras.Input(shape=(None, 1), dtype="float64")
    x = tf.keras.layers.Conv1D(filters = num_filter, kernel_size = ks, padding="same", activation="relu")(in_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)

    for i in range(2,num_layer+1):
        try:
            if i==num_layer:
                x = tf.keras.layers.Conv1D(filters = num_filter, kernel_size = ks, padding="same", activation="relu")(x)
                break
            if i%2 != 0:
                num_filter = num_filter *2
            x = tf.keras.layers.Conv1D(filters = num_filter, kernel_size = ks, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPool1D(2)(x)
            if i in [6, 8, 9]:
                x = tf.keras.layers.Dropout(0.5)(x)
        except ValueError:
            print("model overflow[lr, bs, ks, #layer]: ",[learning_rate, bs, ks, i+1])
            return False
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(64, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation = 'relu')(x)
    outputs = tf.keras.layers.Dense(4, activation = 'softmax')(x)
    
    model = tf.keras.Model(inputs=in_data, outputs=outputs)
    print(model.summary())
    return model
