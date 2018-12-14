import itertools
import pylab as plt
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          index=0,
                          save_png=False,
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    title='Confusion matrix'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    if save_png:
        txt = 'Confusion Matrix:' + str(index) + '.png'
        plt.savefig(txt)
        plt.close()
    else:
        plt.tight_layout()
    
    


'''
from sklearn.metrics import confusion_matrix

test_prediction = model.predict_classes(testData, batch_size=1)
cnf_matrix = confusion_matrix(testL, test_prediction)
plot_confusion_matrix(cnf_matrix, classes=['AF','Noise','Normal','Other'],
                      title='Confusion matrix')
'''