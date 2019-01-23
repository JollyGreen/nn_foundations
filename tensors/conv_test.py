#!/usr/bin/env python

import numpy as np
from scipy.signal import correlate2d
from conv_helpers import *

def corr2d(a,f, padval=0):
	pada=np.pad(a,pad_width=padval, mode='constant', constant_values=0)

	(fh,fw)=f.shape
	(ah,aw)=pada.shape

	oh=ah-fh+1
	ow=aw-fw+1

	o=np.zeros( (oh,ow) )
	for r in range(0,oh):
		for c in range(0,ow):
			o[r,c]=np.dot(f.reshape(-1), pada[r:(r+fh),c:(c+fw)].reshape(-1))
	return o

def corr2d_gemm(a,f, padval=0):
	pada=np.pad(a,pad_width=padval, mode='constant', constant_values=0)

	(fh,fw)=f.shape
	(ah,aw)=pada.shape

	oh=ah-fh+1
	ow=aw-fw+1

	farray=f.reshape(-1)
	flen=farray.shape[0]
	# Nx9 9x1
	tmp=np.zeros( (oh*ow,flen) )
	for r in range(0,oh):
		for c in range(0,ow):
			tmp[r*ow+c,:]=pada[r:(r+fh),c:(c+fw)].reshape(-1)
	result=np.dot(tmp, farray)
	return result.reshape(oh,ow)

def corr2d_multi_channel(a,f, padval=0):
	(ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (padah,padaw,channels) )
	for i in range(0,channels):
		pada[:,:,i]=np.pad(a[:,:,i],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	o=np.zeros( (oh,ow,channels) )
	for channel in range(0,channels):
		for r in range(0,oh):
			for c in range(0,ow):
				o[r,c,channel]=np.dot(f.reshape(-1), pada[r:(r+fh),c:(c+fw), channel].reshape(-1))
	return o

def corr2d_gemm_multi_channel(a,f, padval=0):
	(ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (padah,padaw,channels) )
	for i in range(0,channels):
		pada[:,:,i]=np.pad(a[:,:,i],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	farray=f.reshape(-1)
	flen=farray.shape[0]

	#Nx9 9x1
	tmp=np.zeros( (oh*ow*channels, flen) )
	for channel in range(0,channels):
		for r in range(0,oh):
			for c in range(0,ow):
				tmp[channel*oh*ow+r*ow+c,:]=pada[r:(r+fh),c:(c+fw), channel].reshape(-1)
	result=np.dot(tmp, farray)
	o=np.zeros( (oh,ow,channels) )
	for channel in range(0,channels):
		o[:,:,channel]=result[channel*oh*ow:(channel+1)*oh*ow].reshape(oh,ow)
	return o

def corr3d_tensor(a,f,padval=0):
	(ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (padah,padaw,channels) )
	for i in range(0,channels):
		pada[:,:,i]=np.pad(a[:,:,i],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw,fchannels)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	o=np.zeros( (oh,ow,1) )
	for r in range(0,oh):
		for c in range(0,ow):
			for channel in range(0,channels):
				o[r,c,0]+=np.dot(f[:,:,channel].reshape(-1), pada[r:(r+fh),c:(c+fw), channel].reshape(-1))
	return o

def corr3d_gemm_tensor(a,f, padval=0):
	(ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (padah,padaw,channels) )
	for i in range(0,channels):
		pada[:,:,i]=np.pad(a[:,:,i],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw,fc)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	farray=f.reshape(-1)
	flen=farray.shape[0]

	#Nx27 27x1
	tmp=np.zeros( (oh*ow, flen) )
	for r in range(0,oh):
		for c in range(0,ow):
			tmp[r*ow+c,:]=pada[r:(r+fh),c:(c+fw), :].reshape(-1)
	result=np.dot(tmp, farray)
	return result.reshape(oh,ow,1)

def corr3d_tensor_multi(a,f,padval=0):
	(ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (padah,padaw,channels) )
	for i in range(0,channels):
		pada[:,:,i]=np.pad(a[:,:,i],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw,ic,oc)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	o=np.zeros( (oh,ow,oc) )
	for r in range(0,oh):
		for c in range(0,ow):
			for ochannel in range(0,oc):
				for ichannel in range(0,ic):
					o[r,c,ochannel]+=np.dot(f[:,:,ichannel,ochannel].reshape(-1), pada[r:(r+fh),c:(c+fw), ichannel].reshape(-1))
	return o

def corr3d_gemm_tensor_multi(a,f, padval=0):
	(ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (padah,padaw,channels) )
	for i in range(0,channels):
		pada[:,:,i]=np.pad(a[:,:,i],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw,ic,oc)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1


	flen=fh*fw*ic
	farray=f.reshape(flen,oc)

	#Nx27 27x10
	tmp=np.zeros( (oh*ow, flen) )
	for r in range(0,oh):
		for c in range(0,ow):
			tmp[r*ow+c,:]=pada[r:(r+fh),c:(c+fw), :].reshape(-1)
	result=np.dot(tmp, farray)
	return result.reshape(oh,ow,oc)





#f=np.ones( (2,2) )
f=np.random.randint( 10, size=(3,3) )
a=np.random.randint( 10, size=(28,28) )

print "Single Channel"
print "Scipy built-in Convolution"
pada=np.pad(a,pad_width=1, mode='constant', constant_values=0)
o=correlate2d(pada,f, mode='valid')
print o.shape

print "Brute force 2D Convolution"
o_hat=corr2d(a,f,padval=1)
print o_hat.shape
print (o==o_hat).all()

print "GEMM Convolution"
o_hat_gemm=corr2d_gemm(a,f,padval=1)
print o_hat_gemm.shape
print (o==o_hat_gemm).all()

print "Three Channel"
f=np.random.randint( 10, size=(3,3) )
a=np.random.randint( 10, size=(28,28,3) )
print a.shape

o_hat_multi=corr2d_multi_channel(a,f,padval=1)
print o_hat_multi.shape

print "GEMM Three Channel"
o_hat_gemm_multi=corr2d_gemm_multi_channel(a,f,padval=1)
print o_hat_gemm_multi.shape
print (o_hat_multi==o_hat_gemm_multi).all()

print "3D Tensor"
f=np.random.randint( 10, size=(3,3,3) )

o_hat_3d_tensor=corr3d_tensor(a,f,padval=1)
print o_hat_3d_tensor.shape

print "3D Tensor GEMM"
o_hat_3d_gemm_tensor=corr3d_gemm_tensor(a,f,padval=1)
print o_hat_3d_gemm_tensor.shape
print (o_hat_3d_tensor==o_hat_3d_gemm_tensor).all()

print "3D Tensor Multi"
f=np.random.randint( 10, size=(3,3,3,10) )

o_hat_3d_tensor_multi=corr3d_tensor_multi(a,f,padval=1)
print o_hat_3d_tensor_multi.shape

print "3D GEMM Tensor Multi"

o_hat_3d_gemm_tensor_multi=corr3d_gemm_tensor_multi(a,f,padval=1)
print o_hat_3d_gemm_tensor_multi.shape
print (o_hat_3d_tensor_multi==o_hat_3d_gemm_tensor_multi).all()

print "4D Tensor"
f=np.random.randint( 10, size=(3,3,3,10) )
a=np.random.randint( 10, size=(16,28,28,3) )

o_hat_4d_tensor=corr4d_tensor(a,f,padval=1)
print o_hat_4d_tensor.shape

print "4D GEMM Tensor"
o_hat_4d_gemm_tensor=corr4d_gemm_tensor(a,f,padval=1)
print o_hat_4d_gemm_tensor.shape
print (o_hat_4d_tensor==o_hat_4d_gemm_tensor).all()

print "4D Tensor Flatten Test"
tensor_flatten_test(a)

