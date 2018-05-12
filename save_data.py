import numpy as np
import wfdb as wf
import pandas as pd
np.set_printoptions(suppress=True)
import data_preprocessing as dp

to_9_overlapping = dp.makeData(9, 0.5, 0.2, 0.3, overlap_dot=1350)
trainD, trainL, valD, valL, testD, testL = to_9_overlapping.main()

to_9 = dp.makeData(9, 0.5, 0.2, 0.3)
trainD, trainL, valD, valL, testD, testL = to_9.main()

to_30 = dp.makeData(30, 0.5, 0.2, 0.3)
trainD, trainL, valD, valL, testD, testL = to_30.main()