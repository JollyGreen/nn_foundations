#!/usr/bin/env python

import numpy as np
from scipy.signal import correlate2d
from conv_helpers import *
from pad_helpers import *

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


def im2col_indices_nhwc(zshape):
	(N,H,W,C)=zshape
	FH=2
	FW=2
	stride=2

	OH=H/2
	OW=W/2

	r=[0,0,1,1]
	r=np.tile(r,OW*OH)
	r0=np.repeat(np.arange(0,OH)*stride, FH*FW*OW)
	r=r+r0
	c=[0,1,0,1]
	c=np.tile(c,OH*OW)
	c0=np.tile(np.repeat(np.arange(0,OW)*stride, FH*FW), OH)
	c=c+c0
	r=np.tile(r,C)
	c=np.tile(c,C)
	k=np.repeat(np.arange(0,C), FH*FW*OH*OW)
	return [r,c,k]

def corr4d_gemm_tensor_fast(a,f, padval=0):
	(m,ah,aw,channels)=a.shape

	if (padval > 0):
		pada=pad_fast(a, padval)
	else:
		pada=a

	(fh,fw,ic,oc)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	#print oh,ow
	flen=fh*fw*ic
	#f=f.transpose(2,0,1,3)
	farray=f.reshape(flen,oc)

	#pada=pada.transpose(0,3,1,2)
	#Nx27*m 27x10
	start=time.time()
	tmp=np.zeros( (m*oh*ow, flen) )
	for i in range(0,m):
		for r in range(0,oh):
			for c in range(0,ow):
				tmp[i*oh*ow+r*ow+c,:]=pada[i,r:(r+fh),c:(c+fw),:].reshape(-1)
				#tmp[i*oh*ow+r*ow+c,:]=pada[i,:,r:(r+fh),c:(c+fw)].reshape(-1)
	stop=time.time()
	im2rowtime=stop-start
	#print oh,ow, tmp.shape, farray.shape
	start=time.time()
	result=np.dot(tmp, farray)
	stop=time.time()
	print "gemm time: ", im2rowtime,stop-start
	return result.reshape(m,oh,ow,oc)



#f=np.ones( (2,2) )
f=np.random.randint( 10, size=(3,3) )
a=np.random.randint( 10, size=(28,28) )

print "Single Channel"
print "Scipy built-in valid Convolution"
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

print "Scipy built-in full Convolution"
o=correlate2d(a,f, mode='full')
print o.shape

print "Brute force 2D full Convolution"
o_hat=corr2d(a,f,padval=2)
print o_hat.shape
print (o==o_hat).all()



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

print "4D GEMM rows Tensor"
o_hat_4d_gemm_rows_tensor=corr4d_gemm_rows_tensor(a,f,padval=1)
print o_hat_4d_gemm_rows_tensor.shape
print (o_hat_4d_tensor==o_hat_4d_gemm_rows_tensor).all()

print "4D GEMM cols Tensor"
o_hat_4d_gemm_cols_tensor=corr4d_gemm_cols_tensor(a,f,padval=1)
print o_hat_4d_gemm_cols_tensor.shape
print (o_hat_4d_tensor==o_hat_4d_gemm_cols_tensor).all()

print "4D Tensor Flatten Test"
tensor_flatten_test(a)

