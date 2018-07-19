import spect_preprocessing as sp

trainData, validationData = sp.to_Spectrogram()

np.save('/home/hsiehch/9s/2D_spect/train_data.npy', trainData)
np.save('/home/hsiehch/9s/2D_spect/validate_data.npyv', validationData)
