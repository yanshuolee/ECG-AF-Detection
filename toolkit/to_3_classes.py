# This module is to combine noise and other data
import numpy as np
import time
np.set_printoptions(suppress=True)

# Input: 4 classes label list(1D)
# Output: 3 classes label list(1D)
def to_3_classes(labels, trace_time=False, mode='numpy'):
    
    if trace_time == True:
        # Using numpy
        if mode == 'numpy':
            start = time.time()
            noise_index = np.where(labels==3)
            for i in noise_index:
                labels[i] = 1
            end = time.time()

        # manual
        if mode == 'manual':
            start = time.time()
            for i in range(len(labels)):
                if labels[i] == 3:
                    labels[i] = 1
            end = time.time()
        
        print('Time Elapsed: ', end-start)
        return labels
    
    else:
        # Using numpy
        if mode == 'numpy':
            noise_index = np.where(labels==3)
            for i in noise_index:
                labels[i] = 1

        # manual
        if mode == 'manual':
            for i in range(len(labels)):
                if labels[i] == 3:
                    labels[i] = 1

        return labels
        