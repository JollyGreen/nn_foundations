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

graph=[]
graph.append(LayerInnerProduct(5))
graph.append(LayerInnerProduct(2))
graph.append(LayerSoftmaxLoss())

nn=NN(graph)

print 'forward'
nn.setshapes(x.shape)
yhat=nn.forward(x)
print 'yhat\t',yhat.shape

print 'backward'
nn.backward(y)

