#!/usr/bin/env python

import time
import numpy as np
from maxpool_helpers import fill_nchw
from maxpool_helpers import fill_nhwc

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
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

def im2cols_online_nchw(z,f):
	(N,C,H,W)=z.shape
	(FH,FW,ic,oc)=f.shape

	flen=FH*FW*ic	
	[k,i,j]=get_im2col_indices(z.shape,3,3,0,1)
	cols=z[:,k,i,j]
	cols=cols.transpose(1,2,0).reshape(flen,-1)
	return cols

def im2cols_online_nhwc(z,f):
	(N,H,W,C)=z.shape
	(FH,FW,ic,oc)=f.shape

	flen=FH*FW*ic	
	[k,i,j]=get_im2col_indices((N,C,H,W),3,3,0,1)
	cols=z[:,i,j,k].transpose(1,2,0).reshape(flen,-1)
	return cols

def direct_conv_nhwc(z,f):
	(N,H,W,C)=z.shape
	(FH,FW,ic,oc)=f.shape

	flen=FH*FW*ic	
	[k,i,j]=get_im2col_indices((N,C,H,W),3,3,0,1)
	cols=z[:,i,j,k].transpose(1,2,0).reshape(flen,-1)
	w=f.reshape(-1,oc).transpose()
	return np.dot(w, cols)
N=128
C=32
H=16
W=16

f=np.random.randn(3,3,C,32)
z=fill_nchw( (N,C,H,W) ).astype(int)
z2=fill_nhwc( (N,H,W,C) ).astype(int)

start=time.time()
cols=im2cols_online_nchw(z,f)
stop=time.time()
colstime=stop-start

start=time.time()
cols_nhwc=im2cols_online_nhwc(z2,f)
stop=time.time()
rowstime=stop-start
print colstime, rowstime
print (cols==cols_nhwc).all()
print cols.shape

start=time.time()
result=direct_conv_nhwc(z2,f)
stop=time.time()
convtime=stop-start

print convtime
