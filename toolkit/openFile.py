import wfdb as wf

def openData(ECG_folder_path, filename):
    index = wf.rdsamp(ECG_folder_path + filename)
    record = index[0]
    record.shape = (record.shape[0], )
    return record