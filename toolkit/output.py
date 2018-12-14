from keras import backend as K
import pprint

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing
test = np.random.random((trainData.shape[1], 1))[np.newaxis,...]
layer_outs = [func([test]) for func in functors]
pprint.pprint(layer_outs)