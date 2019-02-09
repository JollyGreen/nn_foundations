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

print 'x:', x.shape
print 'y:', y.shape
print 'xtest:', xtest.shape
print 'ytest:', ytest.shape

graph=[]
graph.append(LayerConv(4))
#graph.append(LayerMaxPool())
graph.append(LayerConv(4))
#graph.append(LayerMaxPool())
graph.append(LayerFlatten())
graph.append(LayerInnerProduct(10))
graph.append(LayerSoftmaxLoss())

numsamples=x.shape[0]
batchsize=128
numpasses=80
numiters=int(np.round(float(numpasses)*(float(numsamples)/float(batchsize))))
itersperpass=2*numiters/numpasses
alpha=0.01

xtmp=x[0:16,:,:,:]
ytmp=y[0:16,:,:,:]

nn=NN(graph)
nn.setshapes(xtmp.shape)

#nn.train(x,y,xtest,ytest, batchsize, itersperpass, numiters, alpha)

yhat=nn.forward(xtmp)
nn.backward(ytmp)
nn.gradcheck(xtmp,ytmp)

