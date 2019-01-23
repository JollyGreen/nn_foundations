#!/usr/bin/env python

import numpy as np

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
		for r in range(0,oh):
			for c in range(0,ow):
				for ochannel in range(0,oc):
					for ichannel in range(0,ic):
						o[i,r,c,ochannel]+=np.dot(f[:,:,ichannel,ochannel].reshape(-1), pada[i,r:(r+fh),c:(c+fw), ichannel].reshape(-1))
	return o

def corr4d_gemm_tensor(a,f, padval=0):
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

	flen=fh*fw*ic
	farray=f.reshape(flen,oc)

	#Nx27*m 27x10
	tmp=np.zeros( (m*oh*ow, flen) )
	for i in range(0,m):
		for r in range(0,oh):
			for c in range(0,ow):
				tmp[i*oh*ow+r*ow+c,:]=pada[i,r:(r+fh),c:(c+fw), :].reshape(-1)
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

