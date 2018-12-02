#!/usr/bin/env python

import numpy as np
from layer_softmax_gradcheck import *
from nn_utils import *

#X=np.zeros((2,100))
#y=np.zeros((2,100))

#[xvals,cvals]=parse_file('data.txt')
[xvals,cvals]=parse_file('circles.txt')

x=np.matrix(xvals)
y=np.matrix(cvals)
m=x.shape[1]
traintest_ratio=0.8
numtrain=int(float(m)*traintest_ratio)
numtest=m-numtrain

xtrain=x[:,0:numtrain]
xtest=x[:,numtrain:]

ytrain=y[:,0:numtrain]
ytest=y[:,numtrain:]

print 'x:', x.shape
print 'y:', y.shape

graph=[]
graph.append(LayerInnerProduct(20))
graph.append(LayerReLU())
#graph.append(LayerSigmoid())
graph.append(LayerInnerProduct(2))
#graph.append(LayerLoss())
graph.append(LayerSoftmaxLoss())

numsamples=xtrain.shape[1]
batchsize=128
numpasses=80
numiters=int(np.round(float(numpasses)*(float(numsamples)/float(batchsize))))
alpha=0.1

nn=NN(graph)
nn.setshapes(x.shape)

nn.train(xtrain, ytrain, xtest, ytest, batchsize, numiters, alpha)
#nn.dumpweights()
