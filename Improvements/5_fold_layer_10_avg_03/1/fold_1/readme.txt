Loss: 0.495
Accuracy: 0.845
F1[A]: 0.821705426356589
F1[~]: 0.638655462184874
F1[N]: 0.8992443324937028
F1[O]: 0.7632027257240206
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_11 (Conv1D)           (None, 8996, 32)          192       
_________________________________________________________________
activation_11 (Activation)   (None, 8996, 32)          0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 8996, 32)          128       
_________________________________________________________________
average_pooling1d_11 (Averag (None, 4498, 32)          0         
_________________________________________________________________
conv1d_12 (Conv1D)           (None, 4494, 32)          5152      
_________________________________________________________________
activation_12 (Activation)   (None, 4494, 32)          0         
_________________________________________________________________
average_pooling1d_12 (Averag (None, 2247, 32)          0         
_________________________________________________________________
conv1d_13 (Conv1D)           (None, 2243, 64)          10304     
_________________________________________________________________
activation_13 (Activation)   (None, 2243, 64)          0         
_________________________________________________________________
average_pooling1d_13 (Averag (None, 1121, 64)          0         
_________________________________________________________________
conv1d_14 (Conv1D)           (None, 1117, 64)          20544     
_________________________________________________________________
activation_14 (Activation)   (None, 1117, 64)          0         
_________________________________________________________________
average_pooling1d_14 (Averag (None, 558, 64)           0         
_________________________________________________________________
conv1d_15 (Conv1D)           (None, 554, 128)          41088     
_________________________________________________________________
activation_15 (Activation)   (None, 554, 128)          0         
_________________________________________________________________
average_pooling1d_15 (Averag (None, 277, 128)          0         
_________________________________________________________________
conv1d_16 (Conv1D)           (None, 273, 128)          82048     
_________________________________________________________________
activation_16 (Activation)   (None, 273, 128)          0         
_________________________________________________________________
average_pooling1d_16 (Averag (None, 136, 128)          0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 136, 128)          0         
_________________________________________________________________
conv1d_17 (Conv1D)           (None, 132, 256)          164096    
_________________________________________________________________
activation_17 (Activation)   (None, 132, 256)          0         
_________________________________________________________________
average_pooling1d_17 (Averag (None, 66, 256)           0         
_________________________________________________________________
conv1d_18 (Conv1D)           (None, 62, 256)           327936    
_________________________________________________________________
activation_18 (Activation)   (None, 62, 256)           0         
_________________________________________________________________
average_pooling1d_18 (Averag (None, 31, 256)           0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 31, 256)           0         
_________________________________________________________________
conv1d_19 (Conv1D)           (None, 27, 512)           655872    
_________________________________________________________________
activation_19 (Activation)   (None, 27, 512)           0         
_________________________________________________________________
average_pooling1d_19 (Averag (None, 13, 512)           0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 13, 512)           0         
_________________________________________________________________
conv1d_20 (Conv1D)           (None, 9, 512)            1311232   
_________________________________________________________________
activation_20 (Activation)   (None, 9, 512)            0         
_________________________________________________________________
average_pooling1d_20 (Averag (None, 4, 512)            0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               262272    
_________________________________________________________________
dropout_8 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 32)                4128      
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 132       
=================================================================
Total params: 2,885,124
Trainable params: 2,885,060
Non-trainable params: 64
_________________________________________________________________
