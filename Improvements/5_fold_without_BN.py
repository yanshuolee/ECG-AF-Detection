import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import itertools
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
import tools_without_BN as tools

training_data, training_label, validation_data, validation_label, validation_cate_label = tools.get_data()

kernel_size = [5]
num_layers = [10]
batch_size = [30]
learning_rate = [0.0001]
overflow_model = 0

def run(bs, path, lr, ks, num_layer):
    fold=1
    for X_train, Y_train, X_val, Y_val, val_cat in zip(training_data,
                                                       training_label,
                                                       validation_data,
                                                       validation_label,
                                                       validation_cate_label):
        print("Fold "+str(fold))
        model = tools.create_model(lr, bs, ks, num_layer)
        inner_path = path+"/fold_"+str(fold)
        if not os.path.exists(inner_path):
            os.makedirs(inner_path)
        
        early_stop = EarlyStopping(patience=20)
        history = model.fit(x = X_train, 
                            y = Y_train,
                            epochs=80,
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
        del model

## main()
import notify
try:
    base_path = "5_fold_layer_10_without_BN/"
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
        print(model.summary())
        if model:
            run(bs, path, lr, ks, num_layer)
        else:
            overflow_model = overflow_model + 1
        del model
        index = index + 1

    print("Training finished! Overflow model: ", overflow_model)
except:
    notify.send('Exception occurred: without BN')
    
notify.send('Finish training! without BN')