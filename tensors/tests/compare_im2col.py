#!/usr/bin/env python

import numpy as np

from maxpool_helpers import fill_nhwc
from maxpool_helpers import fill_nchw

def corr4d_tensor_nhwc(a,f):
	(m,ah,aw,channels)=a.shape
	padval=0
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

def im2col_indices_nhwc(z,fshape):
	(N,H,W,C)=z.shape
	(FH,FW)=fshape

	flen=FH*FW*C
	[k,i,j]=get_im2col_indices((N,C,H,W),3,3,0,1)
	cols=z[:,i,j,k].transpose(1,2,0).reshape(flen,-1)
	return cols

def unrollW(f):
	(FH,FW,IC,OC)=f.shape
	fout=np.zeros( (OC, FH*FW*IC), dtype=int)
	for oc in range(0,OC):
		for ic in range(0,IC):
			for h in range(0,FH):
				for w in range(0,FW):
					fout[oc,ic*FH*FW+h*FW+w]=f[h,w,ic,oc]
	return fout

def corr4d_tensor_indices_nhwc(a,f):
	(N,H,W,IC)=a.shape
	(FH,FW,IC,OC)=f.shape
	OH=H-FH+1
	OW=W-FW+1

	cols=im2col_indices_nhwc(a, (FH,FW))
	w=f.transpose(3,2,0,1).reshape(OC,-1)
	print f.shape
	w2=unrollW(f.astype(int))
	print (w==w2).all()
	print w2
	print w.shape, cols.shape
	result=np.dot(w,cols)
	print result.shape
	print result
	return result.reshape(OC,OH,OW,N).transpose(3,1,2,0)

def im2col_brute_nhwc(z, fshape):
	(N,H,W,C)=z.shape
	(FH,FW)=fshape

	OH=H-FH+1
	OW=W-FW+1

	o=np.zeros( (FH*FW*C, N*OH*OW) )

	for r in range(0,OH):
		for c in range(0,OW):
			for n in range(0,N):
				for channel in range(0,C):
					o[channel*FH*FW:(channel+1)*FH*FW,r*OW*N+c*N+n]=z[n, r:(r+FH), c:(c+FW), channel].reshape(-1)
	return o


N=128
H=28
W=28
C=2
OC=3

z=fill_nhwc( (N,H,W,C) ).astype(int)
W=np.zeros( (3,3,C,OC), dtype=int)

W=np.arange(0,3*3*C*OC).reshape(OC,C,3,3).transpose(2,3,1,0)

cols_brute=im2col_brute_nhwc( z, (3,3) ).astype(int)
cols_indices=im2col_indices_nhwc( z, (3,3) ).astype(int)
print z[0,:,:,0]
print z[0,:,:,1]
print z[1,:,:,0]
print z[1,:,:,1]
print cols_brute
print cols_indices

print "Brute force cols match indices cols", (cols_brute==cols_indices).all()
print W[:,:,0,0]
print W[:,:,1,0]
print W[:,:,0,1]
print W[:,:,1,1]

corr=corr4d_tensor_nhwc(z,W).astype(int)
print corr.shape
print corr[0,:,:,0]
print corr[0,:,:,1]
print corr[0,:,:,2]
corr2=corr4d_tensor_indices_nhwc(z,W)
print (corr==corr2).all()
