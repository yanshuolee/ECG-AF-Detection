import os
import itertools
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import f1_score
import tools_with_GMP as tools
import plot_confusion_matrix_Copy1 as plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
np.set_printoptions(suppress=True)

training_data, training_label, validation_data, validation_label, validation_cate_label = tools.get_data()

ks = 5
num_layer = 5
bs = 30
lr = 0.0005

print("Start~")

def run(bs, lr, ks, num_layer):
    fold=1
    losses = []
    for index, (X_train, Y_train, X_val, Y_val, val_cat) in enumerate(zip(training_data,
                                                       training_label,
                                                       validation_data,
                                                       validation_label,
                                                       validation_cate_label)):

        model = tools.create_model(lr, bs, ks, num_layer)
        optimizer = tf.keras.optimizers.Adam(lr = lr)
        ##early_stop = EarlyStopping(patience=20)
        epochs = 100
        
        for epoch in range(1, epochs+1, 1):
            print("Epoch: {}".format(epoch))
            epoch_losses = []
            
            prog_bar = tf.keras.utils.Progbar(X_train.shape[0], verbose=1)
            for ind, (input_data, input_label) in enumerate(zip(X_train, Y_train)):
                
                with tf.GradientTape() as tape:
                    input_data = input_data.reshape((1, input_data.shape[0], 1))
                    input_label = input_label.reshape((1, input_label.shape[0]))
                    logits = model(input_data)
                    loss_value = tf.keras.losses.categorical_crossentropy(input_label, logits)
                    epoch_losses.append(float(loss_value))
                
                prog_bar.add(1, values=[("train loss", float(loss_value))])
                
                # update weights
                if (ind+1)%bs == 0:
                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            avg_epoch_loss = sum(epoch_losses) / (1.0 * len(epoch_losses))
            
            print("{}: {}".format(epoch, avg_epoch_loss))
            losses.append(avg_epoch_loss)

#         history = model.fit(x = X_train, 
#                             y = Y_train,
#                             epochs=100,
#                             validation_data=(X_val, Y_val),
#                             callbacks=[early_stop],
#                             batch_size=bs, 
#                             verbose=1)
#         model.save('model_with_GMP_{}.h5'.format(fold))
#         model = load_model('model_with_GMP.h5')
        
#         evaluation = model.evaluate(x = X_val, y = Y_val)
#         validation_prediction = model.predict_classes(X_val, batch_size=bs)
#         score = f1_score(val_cat, validation_prediction, average=None)
#         print(score)
        
        fold = fold + 1
        losses.append("||")
    
    return losses
        
#         test_prediction = model.predict_classes(X_val, batch_size=1)
#         cnf_matrix = confusion_matrix(val_cat, test_prediction)
#         plot_confusion_matrix.plot_confusion_matrix(cnf_matrix, classes=['AF','Noise','Normal','Other'], save_png=True)
        
#         return X_val, val_cat, validation_prediction
    
losses = run(bs, lr, ks, num_layer)