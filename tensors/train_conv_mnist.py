#!/usr/bin/env python

import numpy as np
from layers import *
from nn_utils import *

[traindata, trainlabels]=parse_mnist_train("../mnist/train-images-idx3-ubyte", "../mnist/train-labels-idx1-ubyte", 60000)
[testdata, testlabels]=parse_mnist_train("../mnist/t10k-images-idx3-ubyte", "../mnist/t10k-labels-idx1-ubyte", 10000)

x=traindata.astype(float)/255.0
y=trainlabels.astype(float)

xtest=testdata.astype(float)/255.0
ytest=testlabels.astype(float)

y=np.expand_dims(y,2)
y=np.expand_dims(y,3)

ytest=np.expand_dims(ytest,2)
ytest=np.expand_dims(ytest,3)

x=x[0:6000,:,:,:]
y=y[0:6000,:,:,:]

xtest=xtest[0:1000,:,:,:]
ytest=ytest[0:1000,:,:,:]

print 'x:', x.shape
print 'y:', y.shape
print 'xtest:', xtest.shape
print 'ytest:', ytest.shape

graph=[]
graph.append(LayerConv(32))
graph.append(LayerMaxPool())
graph.append(LayerReLU())
graph.append(LayerConv(64))
graph.append(LayerMaxPool())
graph.append(LayerReLU())
graph.append(LayerDropout(0.8))
graph.append(LayerFlatten())
graph.append(LayerInnerProduct(128))
graph.append(LayerReLU())
graph.append(LayerDropout(0.5))
graph.append(LayerInnerProduct(10))
graph.append(LayerSoftmaxLoss())

numsamples=x.shape[0]
batchsize=128
numpasses=80
numiters=int(np.round(float(numpasses)*(float(numsamples)/float(batchsize))))
itersperpass=numiters/numpasses
alpha=0.003

xtmp=x[0:128,:,:,:]
nn=NN(graph, timing=False)
nn.setshapes(xtmp.shape)

print "ITERS PER PASS: ", itersperpass
nn.train(x,y,xtest,ytest, batchsize, itersperpass, numiters, alpha, timing=False)


