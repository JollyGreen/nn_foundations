#!/usr/bin/env python

import numpy as np
import sys

from conv_helpers import *

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
	r=sigmoid(x)
	return np.multiply(r, (1.0-r))

def prod(x, W, b):
	#return (x*W.getT()+b.getT())
	#return (x*np.transpose(W)+np.transpose(b))
	return (np.dot(x,np.transpose(W))+np.transpose(b))

class LayerConv:
	def __init__(self):
		self.type='conv3x3'
		self.outchannels=1
		self.Wshape=(3,3,1,self.outchannels)
	def setshapes(self, inputshape):
		(m,ah,aw,channels)=inputshape
		self.m=m
		self.zshape=inputshape
		self.ashape=(m,ah,aw,self.outchannels)
		
		self.Wshape=(3,3,1,self.outchannels) # HxWxICxOC
		self.W=np.random.randn(self.Wshape[0],self.Wshape[1],self.Wshape[2],self.Wshape[3])
		self.b=np.random.randn(1,self.outchannels)
		self.deltaB=np.random.randn(1,self.outchannels)
		pass
	def setW(self, W):
		self.W=W
	def setB(self, B):
		self.b=B
	def getW(self):
		return self.W
	def getB(self):
		return self.b
	def forward(self, z):
		#for now, just implement 3x3 stride of 1 convolutions with padding of 1,
		#start with just 1 input channel to 1 output channel.
		padval=1
		(m,zh,zw,channels)=z.shape
		padzh=zh+2*padval
		padzw=zw+2*padval

		self.padz=np.zeros( (m,padzh,padzw,channels) )
		for i in range(0,m):
			for j in range(0,channels):
				self.padz[i,:,:,j]=np.pad(z[i,:,:,j],pad_width=padval, mode='constant', constant_values=0)
		self.a=corr4d_gemm_tensor(self.padz,self.W)
		#print "conv shape",self.a.shape	
		return self.a
	def backward(self, din):
		(m,dh,dw,dchannels)=din.shape
		(m,zh,zw,zchannels)=self.padz.shape
		fh=zh-dh+1
		fw=zw-dw+1

		self.deltaW=np.zeros( (fh,fw,zchannels,dchannels) )
		for n in range(0,m):
			for i in range(0,fh):
				for j in range(0,fw):
					for k in range(0,zchannels):
						for l in range(0,dchannels):
							self.deltaW[i,j,k,l]+=np.dot( self.padz[n, i:(i+dh),j:(j+dw), k].reshape(-1), din[n,:,:,l].reshape(-1) )
				
		#print "self.padz.shape",self.padz.shape
		#print "din.shape",din.shape
		#print "deltaW.shape",self.deltaW.shape
		pass
	def update(self, alpha):
		pass

class LayerInnerProduct:
	def __init__(self, numhidden):
		self.type='innerproduct'
		self.numhidden=numhidden
		
		self.first=True

		#momentum weight
		self.gamma=0.9

		#regularization weight
		self.eta=0.0004
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
	def regularization(self):
		regterm=self.eta*np.dot(self.W.reshape(-1),self.W.reshape(-1).transpose())
		return regterm
	def forward(self, z):
		self.z=z
		self.zgemm=np.squeeze(np.squeeze(self.z, axis=3),axis=2)
		#print self.zgemm.shape
		#print self.W.shape
		#print self.b.shape
		self.agemm=prod(self.zgemm,self.W,self.b)
		self.a=np.expand_dims(np.expand_dims(self.agemm, axis=2), axis=3)
		return self.a
	def backward(self, din):
		m=float(self.z.shape[0])
		self.din=din
		self.din_gemm=np.squeeze(np.squeeze(self.din, axis=3),axis=2)

		self.deltaW=(1.0/m)*np.dot(self.din_gemm.transpose(),self.zgemm)
		#L2 regularization
		self.deltaW=self.deltaW+(self.eta/m)*self.W

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

class LayerFlatten:
	def __init__(self):
		self.type='flatten'
		pass
	def setshapes(self, inputshape):
		(m,ah,aw,channels)=inputshape
		self.zshape=inputshape
		self.ashape=(m, ah*aw*channels, 1, 1)
	def forward(self, z):
		(m,ah,aw,channels)=z.shape
		self.z=z
		self.a=self.z.reshape(m,ah*aw*channels,1,1)
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=self.din.reshape( self.z.shape )
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
	def update(self, alpha):
		numlayers=len(self.layers)
		for i in range(0,numlayers):
			layer=self.layers[i]
			if (layer.type=='innerproduct'):
				layer.update(alpha)	
	def costFunc(self, yhat, y):
		cost=0.0
		regterm=0.0
		m=float(y.shape[0])
		numlayers=len(self.layers)
		lastlayer=self.layers[numlayers-1]
		if (lastlayer.type=='loss'):
			cost=lastlayer.costFunc(yhat, y)

		#add in L2 regularization
		for i in range(0,numlayers):
			layer=self.layers[i]
			if (layer.type=='innerproduct'):
				regterm=regterm+0.5*(1.0/m)*layer.regularization()
		return (cost+regterm)
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
	def train(self, xtrain, ytrain, xtest, ytest, batchsize, itersperpass, maxiters, alpha):
		print "maxiters:", maxiters

		for i in range(0, maxiters):
			[batchx,batchy]=self.getminibatch(xtrain,ytrain,batchsize,i)
			yhat=self.forward(batchx)
			self.backward(batchy)

			if (np.mod(i, itersperpass)==0):
				yhatbatch=self.forward(batchx, dropout=False)
				batchcost=self.costFunc(yhatbatch, batchy)
				batchaccuracy=self.accuracy(yhatbatch, batchy)

				yhattest=self.forward(xtest, dropout=False)
				testcost=self.costFunc(yhattest, ytest)
				testaccuracy=self.accuracy(yhattest, ytest)
				print '\ttrain - cost: %7.3f\taccuracy: %7.3f\ttest - cost: %7.3f\taccuracy: %7.3f' % (batchcost, batchaccuracy, testcost, testaccuracy)

				#self.gradcheck(batchx,batchy)

			self.update(alpha)
	def gradcheck(self, x, y):
		epsilon=1e-4
		numlayers=len(self.layers)

		#need to call both forward and backward without dropout, because each forward call randomizes the dropout picks
		#since we need to call forward multiple times to compute the estimated gradient we need dropout turned off so
		#that the estimated gradients match the calculated gradients from the call to "backward"
		#Also, ReLU layers can cause errors in the gradient estimates because the "kink" at zero doesn't have a derivative.
		yhat=self.forward(x,dropout=False)
		self.backward(y, dropout=False)

		for i in range(0, numlayers):
			layer=self.layers[i]
			print "layer.type: ", layer.type
			if (layer.type=='innerproduct' or layer.type=='conv3x3'):
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
					tmpW=(initialParams+eps)[0:numW].reshape(shapeW)
					tmpB=(initialParams+eps)[numW:].reshape(shapeB)
					layer.setW(tmpW)
					layer.setB(tmpB)

					yhat1=self.forward(x,dropout=False)
					cost1=self.costFunc(yhat1, y)

					tmpW=(initialParams-eps)[0:numW].reshape(shapeW)
					tmpB=(initialParams-eps)[numW:].reshape(shapeB)
					layer.setW(tmpW)
					layer.setB(tmpB)

					yhat2=self.forward(x,dropout=False)
					cost2=self.costFunc(yhat2,y)

					result=(cost1-cost2)/(2.0*epsilon)
					numGrad[i]=result
					eps[i]=0.0

				tmpW=(initialParams)[0:numW].reshape(shapeW)
				tmpB=(initialParams)[numW:].reshape(shapeB)
				layer.setW(tmpW)
				layer.setB(tmpB)

				numGradW=numGrad[0:numW].reshape(shapeW)
				numGradB=numGrad[numW:].reshape(shapeB)
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

