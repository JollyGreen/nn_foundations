#!/usr/bin/env python

import numpy as np


def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
	r=sigmoid(x)
	return np.multiply(r, (1.0-r))

def prod(x, W, b):
	return (W.getT()*x+b)


class LayerInnerProduct:
	def __init__(self, numhidden):
		self.type='innerproduct'
		self.numhidden=numhidden
		
		self.first=True
		self.gamma=0.9
	def setshapes(self, inputshape):
		self.zshape=inputshape

		self.Wshape=(self.zshape[0],self.numhidden)
		self.Bshape=(self.numhidden, 1)

		self.ashape=(self.numhidden, self.zshape[1])

		self.W=np.matrix(np.random.normal(size=self.Wshape))
		self.b=np.matrix(np.ones(self.Bshape))

	def setW(self, W):
		self.W=W
	def setB(self, B):
		self.b=B
	def getW(self):
		return self.W
	def getB(self):
		return self.b
	def dumpweights(self):
		print 'W', self.W
		print 'b', self.b
	def forward(self, z):
		self.z=z
		self.a=prod(self.z,self.W,self.b)
		return self.a
	def backward(self, din):
		m=self.z.shape[1]
		self.din=din
		self.deltaW=(1.0/float(m))*self.z*(din.getT())
		self.deltaB=(1.0/float(m))*np.sum(self.din, axis=1)
		self.dout=self.W*(self.din)

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
		self.W=self.W-alpha*self.deltaW
		self.b=self.b-alpha*self.deltaB

class LayerSigmoid:
	def __init__(self):
		self.type='sigmoid'
		pass
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def forward(self, z):
		self.z=z
		self.a=sigmoid(self.z)
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=np.multiply(sigmoid_prime(self.z),din)
		return self.dout

class LayerReLU:
	def __init__(self):
		self.type='relu'
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def forward(self, z):
		self.z=z
		self.a=np.matrix(np.maximum(self.z, 0))
		#print 'relu', self.a.shape
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=np.multiply(np.matrix(np.greater_equal(self.z, 0), dtype=float), din)
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
		self.d=(1.0/self.keepprob)*np.matrix(np.random.rand(self.z.shape[0], self.z.shape[1]) < self.keepprob, dtype=float)
		self.a=np.multiply(self.z, self.d)
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=np.multiply(din, self.d)
		return self.dout

class LayerLoss:
	def __init__(self):
		self.type='loss'
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def costFunc(self, yhat, y):
		print "Squared Error Loss"
		cost=0.0
		numvals=y.shape[1]
		m=numvals
		numfeats=y.shape[0]
		for i in range(0,numvals):
			for j in range(0,numfeats):
				err=0.5*(yhat[j,i]-y[j,i])**2
				cost+=(1.0/float(m))*(err)
		return cost
	def forward(self, z):
		self.z=z
		self.a=z
		return self.a
	def backward(self,y):
		self.err=self.a-y
		return self.err

class LayerSoftmaxLoss:
	def __init__(self):
		self.type='loss'
	def setshapes(self, inputshape):
		self.zshape=inputshape
		self.ashape=self.zshape
	def costFuncSlow(self, yhat, y):
		cost=0.0
		numvals=y.shape[1]
		numclasses=y.shape[0]
		m=numvals
		for i in range(0,numvals):
			for j in range(0, numclasses):
				err=-y[j,i]*np.log(yhat[j,i])
				cost+=(1.0/float(m))*(err)
		return cost
	def costFuncFast(self, yhat, y):
		m=yhat.shape[1]
		cost=-np.sum((1.0/float(m))*np.multiply(y,np.log(yhat)))
		return cost
	def costFunc(self, yhat, y):
		return self.costFuncFast(yhat, y)
	def forward(self, z):
		self.z=z

		#scale by the maxz, could also be a large number to avoid
		#large numbers / overflows from the exponentials
		maxz=np.max(self.z)
		self.z=self.z-maxz

		self.t=np.exp(self.z)
		self.tot=np.sum(self.t, axis=0)
		self.a=self.t/self.tot
		return self.a
	def backward(self,y):
		self.err=self.a-y
		return self.err

class NN:
	def __init__(self,graph):
		self.layers=graph
	def setshapes(self, inputshape):
		numlayers=len(self.layers)
		zshape=inputshape
		for i in range(0, numlayers):
			self.layers[i].setshapes(zshape)
			zshape=self.layers[i].ashape
	def dumpweights(self):
		numlayers=len(self.layers)
		for i in range(0, numlayers):
			if (self.layers[i].type=='innerproduct'):
				self.layers[i].dumpweights()
	def forward(self,x, dropout=True):
		z=x
		numlayers=len(self.layers)
		#print 'Forward'
		for i in range(0, numlayers):
			#print self.layers[i].type
			if ((dropout==False) and (self.layers[i].type=='dropout')):
				pass
			else:
				a=self.layers[i].forward(z)
				z=a
		return a
	def backward(self, din, dropout=True):
		numlayers=len(self.layers)
		#print 'Backward'
		for i in range(numlayers-1, -1, -1):
			#print self.layers[i].type
			if ((dropout==False) and (self.layers[i].type=='dropout')):
				pass
			else:
				dout=self.layers[i].backward(din)
				din=dout
	def costFunc(self, yhat, y):
		cost=0.0
		numlayers=len(self.layers)
		lastlayer=self.layers[numlayers-1]
		if (lastlayer.type=='loss'):
			cost=lastlayer.costFunc(yhat, y)
		return cost
	def update(self, alpha):
		numlayers=len(self.layers)
		for i in range(0,numlayers):
			layer=self.layers[i]
			if (layer.type=='innerproduct'):
				layer.update_momentum(alpha)	
	def binaryaccuracy(self, yhat,y):
		m=y.shape[1]
		val=0
		count=0
		for i in range(0, m):
			if (yhat[0,i] > yhat[1,i]):
				val=1
			else:
				val=0
			if (val==y[0,i]):
				count=count+1
		return (float(count)/float(m))
	def train(self, x, y, batchsize, maxiters, alpha):
		numbatches=x.shape[1]/batchsize
		print "maxiters:", maxiters, "numbatches: ", numbatches
		batchidx=0

		for i in range(0, maxiters):
			batchstart=batchidx*batchsize
			batchend=(batchidx+1)*batchsize
			#print "batchstart: ", batchstart, "batchend", batchend
			batchx=x[:,batchstart:batchend]
			batchy=y[:,batchstart:batchend]

			batchidx=np.mod(i, numbatches)


			yhat=self.forward(batchx)
			self.backward(batchy)

			if (np.mod(i, 100)==0):
				cost=self.costFunc(yhat, batchy)
				accuracy=self.binaryaccuracy(yhat, batchy)
				print 'iter: ', i
				print '\tbatchcost: ', cost, 'batchaccuracy: ', accuracy

				#yhat=self.forward(x)
				#cost=self.costFunc(yhat, y)
				#accuracy=self.binaryaccuracy(yhat, y)
				#print '\tcost: ', cost, 'accuracy: ', accuracy


				#self.gradcheck(batchx,batchy)

				yhat=self.forward(batchx, dropout=False)
				cost=self.costFunc(yhat, batchy)
				accuracy=self.binaryaccuracy(yhat, batchy)
				print '\tcost nodrop:', cost, 'accuracy:', accuracy

			self.update(alpha)
		print batchy[:,0:10]
		print yhat[:,0:10]

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



