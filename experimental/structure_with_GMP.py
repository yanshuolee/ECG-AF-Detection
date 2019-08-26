import os
import itertools
import tensorflow as tf
import early_stopping
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import tools_with_GMP as tools
import plot_confusion_matrix_Copy1 as plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
np.set_printoptions(suppress=True)

training_data, training_label, validation_data, validation_label, validation_cate_label = tools.get_data()

from keras_radam.optimizer_v2 import RAdam
ks = 3
num_layer = 5
bs = 30
lr = 0.0001
epochs = 100

def create_model(learning_rate, bs, ks, num_layer):
    num_filter = 32
    
    in_data = tf.keras.Input(shape=(None, 1), dtype="float64")
    x = tf.keras.layers.Conv1D(filters = num_filter, kernel_size = ks, activation="relu")(in_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)

    for i in range(2,num_layer+1):
        try:
            if i==num_layer:
                x = tf.keras.layers.Conv1D(filters = num_filter, kernel_size = ks, activation="relu")(x)
                break
            if i%2 != 0:
                num_filter = num_filter *2
            x = tf.keras.layers.Conv1D(filters = num_filter, kernel_size = ks, activation="relu")(x)
            x = tf.keras.layers.MaxPool1D(2)(x)
            if i in [6, 8, 9]:
                x = tf.keras.layers.Dropout(0.5)(x)
        except ValueError:
            print("model overflow[lr, bs, ks, #layer]: ",[learning_rate, bs, ks, i+1])
            return False
    
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(32, activation = 'relu')(x)
    outputs = tf.keras.layers.Dense(4, activation = 'softmax')(x)
    
    model = tf.keras.Model(inputs=in_data, outputs=outputs)
    print(model.summary())
    return model

def run(bs, lr, ks, num_layer):
    scores = []
    fold=1
    for index, (X_train, Y_train, X_val, Y_val, val_cat) in enumerate(zip(training_data,
                                                       training_label,
                                                       validation_data,
                                                       validation_label,
                                                       validation_cate_label)):

        X_val, Y_val, val_cat = shuffle(X_val, Y_val, val_cat, random_state=50)
        model = create_model(lr, bs, ks, num_layer)
        optimizer = tf.keras.optimizers.Adam(lr = lr)
        ES = early_stopping.EarlyStopping(patience=6)
        losses = []
        
        for epoch in range(1, epochs+1, 1):
            print("Epoch: {}".format(epoch))
            epoch_losses = []
            
            prog_bar = tf.keras.utils.Progbar(X_train.shape[0])
            prog_bar_val = tf.keras.utils.Progbar(X_val.shape[0])
            train_acc_metric = tf.keras.metrics.Accuracy()
            val_acc_metric = tf.keras.metrics.Accuracy()
            
            X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
            
            for ind, (input_data, input_label) in enumerate(zip(X_train, Y_train)):
                
                with tf.GradientTape() as tape:
                    input_data = input_data.reshape((1, input_data.shape[0], 1))
                    input_label = input_label.reshape((1, input_label.shape[0]))
                    logits = model(input_data)
                    loss_value = tf.keras.losses.categorical_crossentropy(input_label, logits)
                    epoch_losses.append(float(loss_value))
                
                train_acc_metric.update_state(np.argmax(input_label), np.argmax(logits))
                prog_bar.add(1, values=[("train loss", float(loss_value)), ("train accuracy", float(train_acc_metric.result()))])
                
                # update weights using mini-batch mechanism
                if (ind+1)%bs == 0:
                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            avg_epoch_loss = sum(epoch_losses) / (1.0 * len(epoch_losses))
            
            print("{}: {}".format(epoch, avg_epoch_loss))
            losses.append(avg_epoch_loss)
            
            for ind, (input_data, input_label) in enumerate(zip(X_val, Y_val)):
                input_data = input_data.reshape((1, input_data.shape[0], 1))
                input_label = input_label.reshape((1, input_label.shape[0]))
                val_logits = model(input_data)
                val_loss_val = tf.keras.losses.categorical_crossentropy(input_label, val_logits)
                val_acc_metric.update_state(np.argmax(input_label), np.argmax(logits))
                prog_bar_val.add(1, values=[("val loss", float(val_loss_val)), ("val accuracy", float(val_acc_metric.result()))])
        
            print("\n")
            
            ES(float(val_loss_val), model, fold)
            if ES.early_stop:
                print("Early Stopping!")
                break
        
        val_pred_cat = []
        for ind, (input_data, input_label) in enumerate(zip(X_val, Y_val)):
            input_data = input_data.reshape((1, input_data.shape[0], 1))
            input_label = input_label.reshape((1, input_label.shape[0]))
            val_pred = model.predict(input_data)
            val_pred_cat.append(np.argmax(val_pred))
        
        score = f1_score(val_cat, val_pred_cat, average=None)
        scores.append(score)
        print(score)
        
        fold = fold + 1
        
#         cnf_matrix = confusion_matrix(val_cat, val_pred_cat)
#         plot_confusion_matrix.plot_confusion_matrix(cnf_matrix, classes=['AF','Noise','Normal','Other'], save_png=True)
        
        print("\n")
    return scores

scores = run(bs, lr, ks, num_layer)