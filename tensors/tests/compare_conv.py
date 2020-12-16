#!/usr/bin/env python

import time
import numpy as np
from maxpool_helpers import fill_nhwc
from pad_helpers import *
from conv_helpers import *

def corr4d_tensor_nhwc(a,f,padval=0):
	(m,ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	pada=np.zeros( (m,padah,padaw,channels) )
	for i in range(0,m):
		for j in range(0,channels):
			pada[i,:,:,j]=np.pad(a[i,:,:,j],pad_width=padval, mode='constant', constant_values=0)

	(fh,fw,ic,oc)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	o=np.zeros( (m,oh,ow,oc) )
	for i in range(0,m):
		for ochannel in range(0,oc):
			for r in range(0,oh):
				for c in range(0,ow):
					for ichannel in range(0,ic):
						o[i,r,c,ochannel]+=np.dot(f[:,:,ichannel,ochannel].reshape(-1), pada[i,r:(r+fh),c:(c+fw), ichannel].reshape(-1))
	return o

def time_func(strval, f, z, W, padval):
	start=time.time()
	a=f(z,W,padval)
	stop=time.time()
	print strval,stop-start

def local_get_im2col_indices_nchw(x_shape, field_height, field_width, padding=1, stride=1):
	# First figure out what the size of the output should be
	N, C, H, W = x_shape
	assert (H + 2 * padding - field_height) % stride == 0
	assert (W + 2 * padding - field_height) % stride == 0
	out_height = (H + 2 * padding - field_height) / stride + 1
	out_width = (W + 2 * padding - field_width) / stride + 1

	i0 = np.repeat(np.arange(field_height), field_width)
	i0 = np.tile(i0, C)
	i1 = stride * np.repeat(np.arange(out_height), out_width)
	j0 = np.tile(np.arange(field_width), field_height * C)
	j1 = stride * np.tile(np.arange(out_width), out_height)
	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

	return (k, i, j)

def corr4d_gemm_rows_nhwc_wpad(z,W,padval=0):
		#for now, just implement 3x3 stride of 1 convolutions with padding of 1,
		#start with just 1 input channel to 1 output channel.
		padz=pad_fast(z, padval)

		a=corr4d_gemm_rows_tensor(padz,W)

		return a

def corr4d_gemm_cols_nhwc(z,W):
		#for now, just implement 3x3 stride of 1 convolutions with padding of 1,
		#start with just 1 input channel to 1 output channel.
		padz=pad_fast(z, 1)

		a=corr4d_gemm_cols_tensor(padz,W)

		return a


def direct_conv_nhwc(z,f):
	(N,H,W,C)=z.shape
	padz=pad_fast(z, 1)

	(FH,FW,ic,oc)=f.shape

	flen=FH*FW*ic	
	[k,i,j]=local_get_im2col_indices_nchw((N,C,H,W),3,3,1,1)
	cols=padz[:,i,j,k].transpose(1,2,0).reshape(flen,-1)

	w=f.reshape(-1,oc).transpose()
	return np.dot(w, cols).reshape(N,H,W,oc)

def direct_conv_nhwc_test(z,f,padval=0):
	(N,H,W,C)=z.shape
	padz=pad_fast(z, padval)

	(FH,FW,C,OC)=f.shape

	(N,PH,PW,C)=padz.shape

	OH=PH-FH+1
	OW=PW-FW+1

	flen=FH*FW*C
	w=f.transpose(3,2,0,1).reshape(OC,-1)

	[k,i,j]=local_get_im2col_indices_nchw((N,C,PH,PW),3,3,0,1)
	cols=padz[:,i,j,k].transpose(1,2,0).reshape(flen,-1)
	print cols.shape
	return np.dot(w, cols).reshape(OC,OH,OW,N).transpose(3,1,2,0)

def local_corr4d_gemm_cols_tensor(z,f,padval=0):
	(N,H,W,C)=z.shape
	padz=pad_fast(z,padval)

	(FH,FW,C,OC)=f.shape
	(N,PH,PW,C)=padz.shape

	OH=PH-FH+1
	OW=PW-FW+1

	flen=FH*FW*C
	w=f.transpose(3,2,0,1).reshape(OC,-1)

	[k,i,j]=get_im2col_indices_nchw((N,C,H,W),3,3,padval,1)
	cols=padz[:,i,j,k].transpose(1,2,0).reshape(flen,-1)

	return np.dot(w, cols).reshape(OC,OH,OW,N).transpose(3,1,2,0)


N=3
H=6
W=6
C=2

#z1=fill_nhwc( (128,28,28,1) )
z1=np.random.randn(128,28,28,1)
W1=np.random.randn(3,3,1,32)

#z2=fill_nhwc( (128,14,14,32) )
z2=np.random.randn(128,14,14,32)
W2=np.random.randn(3,3,32,64)

time_func("Current Layer1 Forward",corr4d_gemm_rows_nhwc_wpad, z1, W1, 1)
time_func("Current Layer2 Forward",corr4d_gemm_rows_nhwc_wpad, z2, W2, 1)

time_func("Direct Layer1 Forward",corr4d_gemm_cols_tensor, z1, W1, 1)
time_func("Direct Layer2 Forward",corr4d_gemm_cols_tensor, z2, W2, 1)

z3=fill_nhwc( (N,H,W,C) )
W3=fill_nhwc( (3,3,C,1) )

o=corr4d_tensor_nhwc(z3,W3,padval=1)
o2=corr4d_gemm_rows_nhwc_wpad(z3,W3,padval=1)
o3=direct_conv_nhwc_test(z3,W3, padval=1)
o4=corr4d_gemm_cols_tensor(z3,W3,padval=1)
print "Brute match gemm_rows", (o==o2).all()
print "gemm_rows match gemm_cols", (o2==o3).all()
print "gemm_cols match gemm_cols", (o3==o4).all()
