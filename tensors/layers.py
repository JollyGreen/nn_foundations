#!/usr/bin/env python

import numpy as np
import sys
import time

from pad_helpers import *
from conv_helpers import *
from maxpool_helpers import *

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
	def __init__(self,numchannels):
		self.type='conv3x3'
		self.outchannels=numchannels
		self.first=True
		self.firstlayer=False

		#momentum weight
		self.gamma=0.9

		#regularization weight
		self.eta=0.0004
	def setshapes(self, inputshape):
		(m,ah,aw,self.inputchannels)=inputshape
		self.m=m
		self.zshape=inputshape
		self.ashape=(m,ah,aw,self.outchannels)

		fanin=self.inputchannels*3*3
		fanout=self.outchannels
		#s=np.sqrt(2.0/float(fanin+fanout)) #normal


		self.Wshape=(3,3,self.inputchannels,self.outchannels) # HxWxICxOC
		#glorot uniform
		#s=np.sqrt(6.0/float(fanin+fanout)) #uniform
		#self.W=np.random.uniform(low=-s, high=s, size=self.Wshape)

		#xavier weight initialization
		self.W=np.random.randn(self.Wshape[0],self.Wshape[1],self.Wshape[2],self.Wshape[3])*np.sqrt(2.0/(fanin+fanout))

		self.b=np.ones(self.outchannels)
		self.deltaB=np.zeros(self.outchannels)
		pass
	def setfirstlayer(self):
		self.firstlayer=True
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
		#return 0.0
	def forward(self, z):
		#for now, just implement 3x3 stride of 1 convolutions with padding of 1,
		#start with just 1 input channel to 1 output channel.
		self.padz=pad_fast(z, 1)

		#start=time.time()
		self.a=corr4d_gemm_tensor(self.padz,self.W)
		#stop=time.time()
		#gemmtime=stop-start

		#print "times: ",gemmtime
		#self.a=corr4d_tensor(self.padz,self.W)
		for i in range(0,self.outchannels):
			self.a[:,:,:,i]+=self.b[i]

		#print "conv shape",self.a.shape	
		return self.a
	def backward(self, din):
		(m,dh,dw,dchannels)=din.shape
		(m,zh,zw,zchannels)=self.padz.shape

		(tmpfh,tmpfw,ic,oc)=self.W.shape
		rotW=np.zeros( (tmpfh,tmpfw,oc,ic) )
		for k in range(0,dchannels):
			for l in range(0,zchannels):
				rotW[:,:,k,l]=np.rot90(self.W[:,:,l,k], 2)

		self.deltaW=(1.0/float(m))*corr_gemm_tensor_backprop(self.padz,din)
		#L2 regularization
		self.deltaW=self.deltaW+(self.eta/float(m))*self.W

		self.deltaB=np.zeros(self.outchannels)
		for i in range(0,self.outchannels):
			self.deltaB[i]+=(1.0/float(m))*np.sum(din[:,:,:,i])

		if (self.firstlayer == False):
			self.paddin=pad_fast(din, 1)

			self.dout=corr4d_gemm_tensor(self.paddin,rotW)
			#print "self.dout: ",self.dout.shape
			#self.dout=np.random.randn( self.dout.shape[0],self.dout.shape[1],self.dout.shape[2],self.dout.shape[3] )
			return self.dout
		else:
			return []
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

class LayerInnerProduct:
	def __init__(self, numhidden):
		self.type='innerproduct'
		self.numhidden=numhidden
		
		self.first=True
		self.firstlayer=False

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

		fanin=self.numinputfeatures
		fanout=self.numhidden

		#self.W=np.matrix(np.random.normal(size=self.Wshape))
		#xavier weight initialization
		self.W=np.matrix(np.random.randn(self.Wshape[0],self.Wshape[1]), dtype=float)*np.sqrt(2.0/(fanin+fanout))

		#glorot uniform

		#s=np.sqrt(6.0/float(fanin+fanout)) #uniform
		#self.W=np.matrix(np.random.uniform(low=-s, high=s, size=self.Wshape))

		self.b=np.matrix(np.ones(self.Bshape))
		print '\tW',self.W.shape
		print '\tb',self.b.shape
	def setfirstlayer(self):
		self.firstlayer=True
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
		self.deltaW=self.deltaW+(self.eta/float(m))*self.W

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

class LayerMaxPool:
	def __init__(self):
		self.type='maxpool'
	def setshapes(self, inputshape):
		(m,h,w,channels)=inputshape
		self.zshape=inputshape
		self.ashape=(m,h/2,w/2,channels)
	def forward(self, z):
		[self.a,self.switches]=maxpool_nhwc(z)
		return self.a
	def backward(self, din):
		self.dout=speedup_maxpool_back_nhwc(din,self.switches)
		return self.dout


class LayerMaxPoolBrute:
	def __init__(self):
		self.type='maxpool'
	def setshapes(self, inputshape):
		(m,h,w,channels)=inputshape
		self.zshape=inputshape
		self.ashape=(m,h/2,w/2,channels)
	def forward(self, z):
		(m,h,w,channels)=z.shape
		self.z=z
		self.a=np.zeros( (m,h/2,w/2,channels) )
		self.switches=np.zeros( (m,h/2,w/2,channels), dtype=int)
		for n in range(0,m):
			for c in range(0,channels):
				for i in range(0,h/2):
					for j in range(0,w/2):
						patch=z[n, i*2:(i*2+2), j*2:(j*2+2), c].reshape(-1)
						switch=np.argmax( patch )
						self.switches[n,i,j,c]=switch
						self.a[n,i,j,c]=patch[switch]
		return self.a
	def backward(self, din):
		(m,h,w,channels)=din.shape
		self.dout=np.zeros( (m,h*2,w*2,channels) )
		for n in range(0,m):
			for c in range(0,channels):
				for i in range(0,h*2,2):
					for j in range(0,w*2,2):
						patch=np.zeros( 4 )
						dval=din[n,i/2,j/2,c]
						switchval=self.switches[n,i/2,j/2,c]
						patch[switchval]=dval
						self.dout[n, i:(i+2), j:(j+2), c]=patch.reshape( (2,2) )
		
		return self.dout
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
		self.update()
	def forward(self, z):
		self.z=z
		self.a=np.multiply(self.z, self.d)
		return self.a
	def backward(self, din):
		self.din=din
		self.dout=np.multiply(din, self.d)
		return self.dout
	def update(self):
		self.d=(1.0/self.keepprob)*((np.random.rand(self.zshape[0], self.zshape[1], self.zshape[2], self.zshape[3]) < self.keepprob).astype(float))

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
	def __init__(self,graph, timing=False):
		self.layers=graph
		self.timing=timing

		if (len(self.layers) > 0):
			firstlayer=self.layers[0]
			if (firstlayer.type=='innerproduct' or firstlayer.type=='conv3x3'):
				self.layers[0].setfirstlayer()
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
		if (self.timing):
			print 'Forward Timing'
		for i in range(0, numlayers):
			if ((dropout==False) and (self.layers[i].type=='dropout')):
				pass
			else:
				if (self.timing):
					start=time.time()
					a=self.layers[i].forward(z)
					z=a
					end=time.time()
					print "%12s: %7.2f" % (self.layers[i].type, end-start)
				else:
					a=self.layers[i].forward(z)
					z=a

		return a
	def backward(self, din, dropout=True):
		numlayers=len(self.layers)
		if (self.timing):
			print 'Backward Timing'
		for i in range(numlayers-1, -1, -1):
			#print self.layers[i].type
			if ((dropout==False) and (self.layers[i].type=='dropout')):
				pass
			else:
				if (self.timing):
					start=time.time()
					dout=self.layers[i].backward(din)
					din=dout
					end=time.time()
					print "%12s: %7.2f" % (self.layers[i].type, end-start)
				else:
					dout=self.layers[i].backward(din)
					din=dout
	def update(self, alpha):
		numlayers=len(self.layers)
		for i in range(0,numlayers):
			layer=self.layers[i]
			if (layer.type=='conv3x3'):
				layer.update(alpha)	
			if (layer.type=='innerproduct'):
				layer.update(alpha)	
			if (layer.type=='dropout'):
				layer.update()	
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
			if (layer.type=='conv3x3'):
				regterm=regterm+0.5*(1.0/m)*layer.regularization()
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
	def train(self, xtrain, ytrain, xtest, ytest, batchsize, itersperpass, maxiters, alpha, timing=False):
		print "maxiters:", maxiters
		for i in range(0, maxiters):
			[batchx,batchy]=self.getminibatch(xtrain,ytrain,batchsize,i)

			if (np.mod(i, itersperpass)==0):
				yhatbatch=self.forward(batchx, dropout=False)
				batchcost=self.costFunc(yhatbatch, batchy)
				batchaccuracy=self.accuracy(yhatbatch, batchy)

				yhattest=self.forward(xtest, dropout=False)
				testcost=self.costFunc(yhattest, ytest)
				testaccuracy=self.accuracy(yhattest, ytest)
				print '\n\ttrain - batchcost: %7.3f\taccuracy: %7.3f\ttest - cost: %7.3f\taccuracy: %7.3f' % (batchcost, batchaccuracy, testcost, testaccuracy)

				#self.gradcheck(batchx,batchy)


			if (timing):
				start=time.time()
				yhat=self.forward(batchx)
				stop=time.time()
				forwardtime=stop-start
				
				start=time.time()
				self.backward(batchy)
				stop=time.time()
				backwardtime=stop-start
				print "Forward: %7.2f, Backward: %7.2f" % (forwardtime, backwardtime)
			else:
				sys.stdout.write(".")
				sys.stdout.flush()
				yhat=self.forward(batchx)
				self.backward(batchy)


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
				tmpW=layer.getW()
				tmpB=layer.getB()

				shapeW=tmpW.shape
				shapeB=tmpB.shape
				numW=tmpW.size
				numB=tmpB.size
				#print 'W',layer.getW().reshape(-1)
				#print 'b',layer.getB().reshape(-1)

				initialParams=np.append(np.asarray(layer.getW()).reshape(-1), np.asarray(layer.getB()).reshape(-1))
				print numW,numB

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

