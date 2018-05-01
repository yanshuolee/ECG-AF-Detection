import numpy as np
import wfdb as wf
import pandas as pd
np.set_printoptions(suppress=True)


import data_processing_edit as dp
a = dp.generateData(0.5)
a.modifyDataTo30s()