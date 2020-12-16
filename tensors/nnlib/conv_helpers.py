#!/usr/bin/env python

import time
import numpy as np
from pad_helpers import pad_fast

def get_im2col_indices_nhwc(x_shape, field_height, field_width, padding=1, stride=1):
	# First figure out what the size of the output should be
	N, H, W, C = x_shape
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

def get_im2col_indices_nchw(x_shape, field_height, field_width, padding=1, stride=1):
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

def corr_tensor_backprop(a,b,padval=0):
	(m,ah,aw,ic)=a.shape
	(m,bh,bw,oc)=b.shape
	fh=ah-bh+1
	fw=aw-bw+1

	f=np.zeros( (fh,fw,ic,oc) )
	for n in range(0,m):
		for i in range(0,fh):
			for j in range(0,fw):
				for k in range(0,ic):
					for l in range(0,oc):
						f[i,j,k,l]+=np.dot( a[n,i:(i+bh), j:(j+bw),k].reshape(-1), b[n,:,:,l].reshape(-1))
	return f

def corr_gemm_tensor_backprop(a,b):
	(m,ah,aw,ic)=a.shape
	(m,bh,bw,oc)=b.shape
	fh=ah-bh+1
	fw=aw-bw+1

	agemm=np.zeros( (fh*fw*ic, m*bh*bw) )
	for i in range(0,fh):
		for j in range(0,fw):
			for k in range(0,ic):
				agemm[i*fw*ic+j*ic+k, :]=a[:,i:(i+bh), j:(j+bw),k].reshape(-1)
	bgemm=b.reshape( (m*bh*bw, oc) )
	f=np.dot(agemm, bgemm)

	return f.reshape(fh,fw,ic,oc)

def corr4d_tensor(a,f,padval=0):
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

def corr4d_gemm_cols_tensor(z,f,padval=0):
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



def corr4d_gemm_rows_tensor(a,f, padval=0):
	(m,ah,aw,channels)=a.shape
	padah=ah+2*padval
	padaw=aw+2*padval

	if (padval > 0):
		pada=np.zeros( (m,padah,padaw,channels) )
		for i in range(0,m):
			for j in range(0,channels):
				pada[i,:,:,j]=np.pad(a[i,:,:,j],pad_width=padval, mode='constant', constant_values=0)
	else:
		pada=a

	(fh,fw,ic,oc)=f.shape

	oh=padah-fh+1
	ow=padaw-fw+1

	flen=fh*fw*ic
	farray=f.reshape(flen,oc)

	#Nx27*m 27x10
	tmp=np.zeros( (m*oh*ow, flen) )
	for i in range(0,m):
		for r in range(0,oh):
			for c in range(0,ow):
				tmp[i*oh*ow+r*ow+c,:]=pada[i,r:(r+fh),c:(c+fw),:].reshape(-1)
				#tmp[i*oh*ow+r*ow+c,:]=pada[i,:,r:(r+fh),c:(c+fw)].reshape(-1)
	result=np.dot(tmp, farray)
	return result.reshape(m,oh,ow,oc)

def tensor_flatten_brute_force(a):
	(m,ah,aw,channels)=a.shape
	b=np.zeros( (m,ah*aw*channels, 1, 1) )
	for i in range(0,m):
		for j in range(0,ah):
			for k in range(0,aw):
				for l in range(0,channels):
					b[i, j*aw*channels+k*channels+l,0,0]=a[i,j,k,l]
	return b
	
def tensor_flatten_test(a):
	(m,ah,aw,channels)=a.shape
	flata=a.reshape(m,ah*aw*channels, 1, 1)
	print flata.shape
	flatb=tensor_flatten_brute_force(a)
	print flatb.shape
	print (flata==flatb).all()

	reshapeb=flatb.reshape(m,ah,aw,channels)
	print (a==reshapeb).all()

