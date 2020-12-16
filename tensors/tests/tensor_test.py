#!/usr/bin/env python

from nn_utils import *
from layers import *

[x,y]=parse_file('circles.txt')

x=np.expand_dims(x,2)
x=np.expand_dims(x,3)

y=np.expand_dims(y,2)
y=np.expand_dims(y,3)

print 'x\t', x.shape
print 'y\t', y.shape

m=x.shape[0]
traintest_ratio=0.8
numtrain=int(float(m)*traintest_ratio)
numtest=m-numtrain

xtrain=x[0:numtrain,]
ytrain=y[0:numtrain,]
xtest=x[numtrain:,]
ytest=y[numtrain:,]

graph=[]
graph.append(LayerInnerProduct(10))
graph.append(LayerReLU())
#graph.append(LayerDropout(0.8))
graph.append(LayerInnerProduct(2))
graph.append(LayerSoftmaxLoss())

nn=NN(graph)

nn.setshapes(xtrain.shape)
print 'train'

numsamples=xtrain.shape[0]
batchsize=128
numpasses=80
numiters=int(np.round(float(numpasses)*(float(numsamples)/float(batchsize))))
itersperpass=2*numiters/numpasses
nn.train(xtrain,ytrain,xtest,ytest, batchsize, itersperpass, numiters, 0.3)

