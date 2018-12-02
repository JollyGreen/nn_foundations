#!/usr/bin/env python

import numpy as np
from layer_softmax_gradcheck import *
from nn_utils import *

[traindata, trainlabels]=parse_mnist_train("./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte", 60000)
[testdata, testlabels]=parse_mnist_train("./mnist/t10k-images-idx3-ubyte", "./mnist/t10k-labels-idx1-ubyte", 10000)

x=np.matrix(traindata, dtype=float)/255.0
y=np.matrix(trainlabels, dtype=float)

xtest=np.matrix(testdata, dtype=float)/255.0
ytest=np.matrix(testlabels, dtype=float)

print 'x:', x.shape
print 'y:', y.shape
m=x.shape[1]

graph=[]
graph.append(LayerInnerProduct(500))
graph.append(LayerReLU())
graph.append(LayerDropout(0.5))
graph.append(LayerInnerProduct(300))
graph.append(LayerReLU())
graph.append(LayerDropout(0.8))
#graph.append(LayerSigmoid())
graph.append(LayerInnerProduct(10))
#graph.append(LayerLoss())
graph.append(LayerSoftmaxLoss())

numsamples=x.shape[1]
batchsize=128
numpasses=80
numiters=int(np.round(float(numpasses)*(float(numsamples)/float(batchsize))))
alpha=0.1

nn=NN(graph)
nn.setshapes(x.shape)

nn.train(x, y, xtest, ytest, batchsize, numiters, alpha)
#nn.dumpweights()
