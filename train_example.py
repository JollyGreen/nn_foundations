#!/usr/bin/env python

import numpy as np
from layer_softmax_gradcheck import *
from nn_utils import *
import matplotlib.pyplot as plt

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

yhattest=nn.forward(xtest, dropout=False)

def confusion_matrix(ypred, yactual):
	# True Positive,  predicted YES, actual YES
	# False Positive, predicted YES, actual  NO
	# False Negative, predicted  NO, actual YES
	# True Negative,  predicted  NO, actual  NO
	m=len(ypred)
	TP=0
	FP=0
	FN=0
	TN=0
	for i in range(0,m):
		if ((ypred[i]==1) and (yactual[i]==1)):
			TP=TP+1
		if ((ypred[i]==1) and (yactual[i]==0)):
			FP=FP+1
		if ((ypred[i]==0) and (yactual[i]==1)):
			FN=FN+1
		if ((ypred[i]==0) and (yactual[i]==0)):
			TN=TN+1
	ACC=float(TP+TN)/float(TP+TN+FP+FN)
	TPR=float(TP)/float(TP+FN)
	FPR=float(FP)/float(FP+TN)
	print "%3d, %3d, %3d, %3d" % (TP,TN,FP,FN),
	return [ACC,TPR,FPR]

def generate_binary_ROC(yhat, y):
	m=y.shape[1]
	roc_data={}
	for threshint in range(0,20):
		threshold=float(threshint+1)/20.0
		ythresh=np.matrix(np.greater_equal(yhat[0,:], threshold), dtype=float)
		[acc, tpr, fpr]=confusion_matrix(np.array(ythresh).flatten(), np.array(y).flatten())
		print "\t%7.3f, %7.3f, %7.3f, %7.3f" % (threshold, acc, tpr, fpr)
		try:
			if (tpr > roc_data[fpr]):
				roc_data[fpr]=tpr
		except:
			roc_data[fpr]=tpr

	x=[]
	y=[]
	for key in roc_data:
		val=roc_data[key]
		x.append(key)
		y.append(val)

	plt.plot(x,y, '+')
	plt.ylim(0.0,1.0)
	plt.xlim(0.0,1.0)
	plt.show()

generate_binary_ROC(yhattest, ytest)


#nn.dumpweights()
