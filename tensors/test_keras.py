#!/usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from nn_utils import *


[traindata, trainlabels]=parse_mnist_train("../mnist/train-images-idx3-ubyte", "../mnist/train-labels-idx1-ubyte", 60000)
[testdata, testlabels]=parse_mnist_train("../mnist/t10k-images-idx3-ubyte", "../mnist/t10k-labels-idx1-ubyte", 10000)

x=traindata.astype(float)/255.0
y=trainlabels.astype(float)

xtest=testdata.astype(float)/255.0
ytest=testlabels.astype(float)

#y=np.expand_dims(y,2)
#y=np.expand_dims(y,3)

#ytest=np.expand_dims(ytest,2)
#ytest=np.expand_dims(ytest,3)

x=x[0:6000,:,:,:]
y=y[0:6000,:]

xtest=xtest[0:1000,:,:,:]
ytest=ytest[0:1000,:]

print 'x:', x.shape
print 'y:', y.shape
print 'xtest:', xtest.shape
print 'ytest:', ytest.shape

input_shape=(28,28,1)

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dropout(0.2))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.fit(x, y, epochs=80, batch_size=128)
[score,acc] = model.evaluate(xtest, ytest, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

