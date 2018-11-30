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

print 'x:', x.shape
print 'y:', y.shape
m=x.shape[1]

graph=[]
graph.append(LayerInnerProduct(m, 20))
graph.append(LayerReLU())
graph.append(LayerInnerProduct(m, 2))
#graph.append(LayerLoss())
graph.append(LayerSoftmaxLoss())

nn=NN(m, graph)
nn.setshapes(x.shape)

nn.train(x, y, 160, 1.0)
nn.dumpweights()
