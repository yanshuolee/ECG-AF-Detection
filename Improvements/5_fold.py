import os
import itertools
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
import tools_stable as tools
from keras import backend as K
import gc
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

training_data, training_label, validation_data, validation_label, validation_cate_label = tools.get_data()

# kernel_size = [7]
kernel_size = [3, 5, 7]
num_layers = [8, 9, 10, 11]
#num_layers = [8]
batch_size = [30, 50, 70, 90, 120]
learning_rate = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
overflow_model = 0

def run(bs, path, lr, ks, num_layer, model):
    fold=1
    for X_train, Y_train, X_val, Y_val, val_cat in zip(training_data,
                                                       training_label,
                                                       validation_data,
                                                       validation_label,
                                                       validation_cate_label):
        print("Fold "+str(fold))
        inner_path = path+"/fold_"+str(fold)
        if not os.path.exists(inner_path):
            os.makedirs(inner_path)
        print(model.summary())
        early_stop = EarlyStopping(patience=20)
        history = model.fit(x = X_train, 
                            y = Y_train,
                            epochs=10,
                            validation_data=(X_val, Y_val),
                            callbacks=[early_stop],
                            batch_size=bs, 
                            verbose=0)
        evaluation = model.evaluate(x = X_val, y = Y_val)
        validation_prediction = model.predict_classes(X_val, batch_size=bs)
        score = f1_score(val_cat, validation_prediction, average=None)
        
        tools.show_plot(inner_path, history)
        tools.write_file(inner_path+"/readme.txt", evaluation, score, model)
        fold = fold + 1

## main()
base_path = "5_fold_test/"
if not os.path.exists(base_path):
    os.makedirs(base_path)
index=1
with open(base_path+"Cartesian_product.txt", "w+") as writer:
    for ks, num_layer, bs, lr in itertools.product(kernel_size, num_layers, batch_size, learning_rate):
        writer.write(str(index)+":"+"\n")
        writer.write('ks={}, num_layer={}, bs={}, lr={}'.format(ks, num_layer, bs, lr)+"\n")
        index = index + 1
    del index

index=1
for ks, num_layer, bs, lr in itertools.product(kernel_size, num_layers, batch_size, learning_rate):
    print("Index "+str(index))
    path = base_path+str(index)
    if not os.path.exists(path):
        os.makedirs(path)
    
    model = tools.create_model(lr, bs, ks, num_layer)
    if model:
        run(bs, path, lr, ks, num_layer, model)
    else:
        overflow_model = overflow_model + 1
    K.clear_session()
    del model
    gc.collect()
    index = index + 1
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

print("Training finished! Overflow model: ", overflow_model)