#!/usr/bin/env python

import numpy as np
import sys

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
	r=sigmoid(x)
	return np.multiply(r, (1.0-r))

def prod(x, W, b):
	return (x*W.getT()+b.getT())

class LayerInnerProduct:
	def __init__(self, numhidden):
		self.type='innerproduct'
		self.numhidden=numhidden
		
		self.first=True
		self.gamma=0.9
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.m=self.zshape[0]
		self.numinputfeatures=self.zshape[1]

		self.Wshape=(self.numhidden, self.numinputfeatures)
		self.Bshape=(self.numhidden,1)

		self.ashape=(self.m,self.numhidden,1,1)

		#self.W=np.matrix(np.random.normal(size=self.Wshape))
		#xavier weight initialization
		self.W=np.matrix(np.random.randn(self.Wshape[0],self.Wshape[1]), dtype=float)*np.sqrt(2.0/(self.Wshape[1]))

		self.b=np.matrix(np.ones(self.Bshape))
		print '\tW',self.W.shape
		print '\tb',self.b.shape
	def setW(self, W):
		self.W=W
	def setB(self, B):
		self.b=B
	def getW(self):
		return self.W
	def getB(self):
		return self.b
	def forward(self, z):
		self.z=z
		self.zgemm=np.squeeze(np.squeeze(self.z, axis=3),axis=2)
		self.agemm=prod(self.zgemm,self.W,self.b)
		self.a=np.expand_dims(np.expand_dims(self.agemm, axis=2), axis=3)
		return self.a
	def backward(self, din):
		m=self.z.shape[0]
		self.din=din
		self.din_gemm=np.squeeze(np.squeeze(self.din, axis=3),axis=2)

		self.deltaW=(1.0/float(m))*np.dot(self.din_gemm.transpose(),self.zgemm)
		self.deltaB=(1.0/float(m))*np.sum(self.din_gemm, axis=0).transpose()
		self.deltaB=np.expand_dims(self.deltaB, axis=1)
		#print '\tdeltaW',self.deltaW.shape
		#print '\tdeltaB',self.deltaB.shape
		self.dout_gemm=(self.din_gemm)*self.W
		self.dout=np.expand_dims(np.expand_dims(self.dout_gemm, axis=2), axis=3)
		
		return self.dout
	def update_momentum(self, alpha):
		if (self.first==True):
			self.first=False
			self.W=self.W-alpha*self.deltaW
			self.b=self.b-alpha*self.deltaB

			self.updateW=self.deltaW
			self.updateB=self.deltaB
		else:
			self.updateW=self.gamma*self.updateW+self.deltaW
			self.updateB=self.gamma*self.updateB+self.deltaB

			self.W=self.W-alpha*self.updateW
			self.b=self.b-alpha*self.updateB
	def update(self, alpha):
		self.update_momentum(alpha)
		#self.W=self.W-alpha*self.deltaW
		#self.b=self.b-alpha*self.deltaB

class LayerReLU:
	def __init__(self):
		self.type='relu'
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def forward(self, z):
		self.z=z
		self.a=np.maximum(self.z, 0)
		#print 'relu', self.a.shape
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=np.multiply(np.greater_equal(self.z, 0), din)
		return self.dout

class LayerDropout:
	def __init__(self, keepprob):
		self.type='dropout'
		self.keepprob=keepprob
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def forward(self, z):
		self.z=z
		self.d=(1.0/self.keepprob)*((np.random.rand(self.z.shape[0], self.z.shape[1], self.z.shape[2], self.z.shape[3]) < self.keepprob).astype(float))
		self.a=np.multiply(self.z, self.d)
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=np.multiply(din, self.d)
		return self.dout

class LayerSoftmaxLoss:
	def __init__(self):
		self.type='loss'
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def forward(self, z):
		self.z=z
		self.zgemm=np.squeeze(np.squeeze(self.z, axis=3),axis=2)

		#scale by the maxz, could also be a large number to avoid
		#large numbers / overflows from the exponentials
		maxz=np.max(self.zgemm)
		self.zgemm=self.zgemm-maxz

		self.t=np.exp(self.zgemm)
		self.tot=np.sum(self.t, axis=1)
		self.tot=np.expand_dims(self.tot,1)
		self.agemm=self.t/self.tot
		self.a=np.expand_dims(np.expand_dims(self.agemm, axis=2), axis=3)
		return self.a
	def backward(self,y):
		self.err=self.a-y
		return self.err
	def costFuncFast(self, yhat, y):
		m=yhat.shape[0]
		cost=-np.sum((1.0/float(m))*np.multiply(y,np.log(yhat)))
		return cost
	def costFunc(self, yhat, y):
		return self.costFuncFast(yhat, y)

class NN:
	def __init__(self,graph):
		self.layers=graph
	def setshapes(self, inputshape):
		numlayers=len(self.layers)
		zshape=inputshape
		for i in range(0, numlayers):
			print self.layers[i].type
			print '\t',zshape
			self.layers[i].setshapes(zshape)
			zshape=self.layers[i].ashape
			print '\t',zshape
	def forward(self,x, dropout=True):
		z=x
		numlayers=len(self.layers)
		#print 'Forward'
		for i in range(0, numlayers):
			if ((dropout==False) and (self.layers[i].type=='dropout')):
				pass
			else:
				a=self.layers[i].forward(z)
				z=a
		return a
	def backward(self, din):
		numlayers=len(self.layers)
		#print 'Backward'
		for i in range(numlayers-1, -1, -1):
			#print self.layers[i].type
			dout=self.layers[i].backward(din)
			din=dout
	def update(self, alpha):
		numlayers=len(self.layers)
		for i in range(0,numlayers):
			layer=self.layers[i]
			if (layer.type=='innerproduct'):
				layer.update(alpha)	
	def costFunc(self, yhat, y):
		cost=0.0
		numlayers=len(self.layers)
		lastlayer=self.layers[numlayers-1]
		if (lastlayer.type=='loss'):
			cost=lastlayer.costFunc(yhat, y)
		return cost
	def accuracy(self, yhat,y):
		m=y.shape[0]
		count=0
		for i in range(0, m):
			maxyhat=np.argmax(yhat[i,])
			if (y[i,maxyhat]==1.0):
				count=count+1
		return (float(count)/float(m))
	def getminibatch(self, x, y, batchsize, i):
		samples=x.shape[0]
		batchstartidx=np.mod(i*batchsize, samples)

		if ((batchstartidx+batchsize) <= samples):
			batchendidx=batchstartidx+batchsize
			batchx=x[batchstartidx:batchendidx,]
			batchy=y[batchstartidx:batchendidx,]
		else:
			#print 'split batch'
			numfirsthalf=samples-batchstartidx
			batchx=x[batchstartidx:,]
			batchy=y[batchstartidx:,]
			#print '\tbatchx.shape', batchx.shape

			endidx=np.mod(batchstartidx+batchsize, samples)
			lasthalfx=x[0:endidx,]
			lasthalfy=y[0:endidx,]
			#print '\tlasthalfx.shape', lasthalfx.shape

			batchx=np.concatenate((batchx,lasthalfx), axis=0)
			batchy=np.concatenate((batchy,lasthalfy), axis=0)
			#print '\tbatchx.shape', batchx.shape
		return [batchx,batchy]
	def train(self, xtrain, ytrain, xtest, ytest, batchsize, maxiters, alpha):
		print "maxiters:", maxiters

		for i in range(0, maxiters):
			[batchx,batchy]=self.getminibatch(xtrain,ytrain,batchsize,i)
			yhat=self.forward(batchx)
			self.backward(batchy)

			if (np.mod(i, 100)==0):
				yhatbatch=self.forward(batchx, dropout=False)
				batchcost=self.costFunc(yhatbatch, batchy)
				batchaccuracy=self.accuracy(yhatbatch, batchy)

				yhattest=self.forward(xtest, dropout=False)
				testcost=self.costFunc(yhattest, ytest)
				testaccuracy=self.accuracy(yhattest, ytest)
				print '\ttrain - cost: %7.2f\taccuracy: %7.2f\ttest - cost: %7.2f\taccuracy: %7.2f' % (batchcost, batchaccuracy, testcost, testaccuracy)

				#self.gradcheck(batchx,batchy)

			self.update(alpha)
	def gradcheck(self, x, y):
		epsilon=1e-4
		numlayers=len(self.layers)
		for i in range(0, numlayers):
			layer=self.layers[i]
			if (layer.type=='innerproduct'):
				print 'Layer Gradcheck'
				shapeW=layer.getW().shape
				shapeB=layer.getB().shape
				numW=shapeW[0]*shapeW[1]
				numB=shapeB[0]*shapeB[1]

				#print 'W',layer.getW().reshape(-1)
				#print 'b',layer.getB().reshape(-1)

				initialParams=np.append(np.asarray(layer.getW()).reshape(-1), np.asarray(layer.getB()).reshape(-1))

				eps=np.zeros(initialParams.shape).reshape(-1)
				numGrad=np.zeros(initialParams.shape).reshape(-1)
				numParams=len(eps)
				#print numParams

				#print 'init', initialParams
				#print 'eps', eps
				for i in range(0,numParams):
					eps[i]=epsilon
					tmpW=np.matrix((initialParams+eps)[0:numW].reshape(shapeW))
					tmpB=np.matrix((initialParams+eps)[numW:].reshape(shapeB))
					layer.setW(tmpW)
					layer.setB(tmpB)

					yhat1=self.forward(x)
					cost1=self.costFunc(yhat1, y)

					tmpW=np.matrix((initialParams-eps)[0:numW].reshape(shapeW))
					tmpB=np.matrix((initialParams-eps)[numW:].reshape(shapeB))
					layer.setW(tmpW)
					layer.setB(tmpB)

					yhat2=self.forward(x)
					cost2=self.costFunc(yhat2,y)

					result=(cost1-cost2)/(2.0*epsilon)
					numGrad[i]=result
					eps[i]=0.0

				tmpW=np.matrix((initialParams)[0:numW].reshape(shapeW))
				tmpB=np.matrix((initialParams)[numW:].reshape(shapeB))
				layer.setW(tmpW)
				layer.setB(tmpB)

				self.forward(x)

				numGradW=np.matrix(numGrad[0:numW].reshape(shapeW))
				numGradB=np.matrix(numGrad[numW:].reshape(shapeB))
				deltas=np.append(np.asarray(layer.deltaW.reshape(-1)), np.asarray(layer.deltaB.reshape(-1)))
				#print 'deltaW', layer.deltaW
				#print 'numGradW', numGradW
				#print 'deltaB', layer.deltaB
				#print 'numGradB', numGradB
				#print 'deltas', deltas
				#print 'numGrad', numGrad

				scoreW=np.linalg.norm(layer.deltaW-numGradW)/np.linalg.norm(layer.deltaW+numGradW)
				scoreB=np.linalg.norm(layer.deltaB-numGradB)/np.linalg.norm(layer.deltaB+numGradB)
				score=np.linalg.norm(deltas-numGrad)/np.linalg.norm(deltas+numGrad)
				print '\tgradcheck score ', 'W: ',scoreW,'B: ', scoreB, 'C: ', score

