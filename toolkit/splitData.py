import numpy as np

def splitData(newData, newLabel, LABEL_TOTAL_COUNT, percentageForTraining, percentageForValidation):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []

    global labels_amount
    labels_amount = LABEL_TOTAL_COUNT
    
    dataIndex = getDataPosition(percentageForTraining, percentageForValidation)
    
    for i in dataIndex:
        for j in range(i[0], i[3]):
            if j < i[1]:
                train_data.append(newData[j])
                train_label.append(newLabel[j])
            elif j >= i[1] and j < i[2]:
                val_data.append(newData[j])
                val_label.append(newLabel[j])
            else:
                test_data.append(newData[j])
                test_label.append(newLabel[j])
        

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    val_data = np.array(val_data)
    val_label = np.array(val_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    return train_data, train_label, val_data, val_label, test_data, test_label

def getDataPosition(percentageForTraining, percentageForValidation):
    amount_for_training = [int(i*percentageForTraining)+1 for i in labels_amount]
    if percentageForValidation != 0:
        amount_for_val = [int(i*percentageForValidation)+1 for i in labels_amount]
    else:
        amount_for_val = [0,0,0,0]
    startPoint = 0
    dataPosition = []

    for i in range(len(labels_amount)):
        trainPoint = amount_for_training[i]+startPoint
        validationPoint = amount_for_training[i]+amount_for_val[i]+startPoint
        endPoint = labels_amount[i]+startPoint
        dataPosition.append([startPoint, trainPoint, validationPoint, endPoint])
        startPoint += labels_amount[i]
    
    print('split train data index: ', dataPosition)
    return dataPosition
