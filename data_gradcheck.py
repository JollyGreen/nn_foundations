#!/usr/bin/env python

import numpy as np
from layer_softmax_gradcheck import *
from nn_utils import *

[xvals,cvals]=parse_file('data.txt')

x=np.matrix(xvals)
y=np.matrix(cvals)

m=1000
x=np.matrix(np.random.normal(size=(2,m)))
y=np.matrix(np.ones((2,m)))
for i in range(0,m/4):
	y[0,i]=0
for i in range(0,m):
	y[1,i]=1.0-y[0,i]


print 'x:', x.shape
print 'y:', y.shape
print 'm:', m

print x[:,0:20]
print y[:,0:20]
graph=[]
graph.append(LayerInnerProduct(m, 3))
graph.append(LayerSigmoid())
graph.append(LayerInnerProduct(m, 2))
graph.append(LayerSigmoid())
#graph.append(LayerLoss())
graph.append(LayerSoftmaxLoss())

nn=NN(m, graph)
nn.setshapes(x.shape)

yhat=nn.forward(x)
nn.backward(y)

nn.gradcheck(x,y)

