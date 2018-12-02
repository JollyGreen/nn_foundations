#!/usr/bin/env python

import numpy as np
from layer_softmax_gradcheck import *
from nn_utils import *

import matplotlib.pyplot as plt

[traindata, trainlabels]=parse_mnist_train("./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte", 20)

image=traindata[:,2].reshape(28,28)

print traindata.shape
print image
print trainlabels
plt.imshow(image)

plt.show()
