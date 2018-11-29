#!/usr/bin/env python

import numpy as np
from layer_softmax_gradcheck import *
from nn_utils import *


#        O 
#    W1 / \ W2
# X -> O-O-O -> Y
#       \ /
#        O
m=1000
x=np.matrix(np.random.normal(size=(2,m)))+5.0

#squared error
#y=np.matrix(np.ones((1,m)))

#softmax
y=np.matrix(np.ones((2,m)))
for i in range(0,m):
	y[1,i]=0

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

err=yhat-y
print err.shape
nn.backward(y)

nn.gradcheck(x, y)

#nn.train(x, y)
#print yhat

