#!/usr/bin/env python

import numpy as np

def pad_brute(z, padval=0):
	(N,H,W,C)=z.shape

	padzh=H+2*padval
	padzw=W+2*padval

	padz=np.zeros( (N,padzh,padzw,C) )
	for i in range(0,N):
		for j in range(0,C):
			padz[i,:,:,j]=np.pad(z[i,:,:,j],pad_width=padval, mode='constant', constant_values=0)
	return padz
def pad_fast(z, padval=0):
	npad=((0,0),(padval,padval),(padval,padval),(0,0))
	return np.pad(z, pad_width=npad, mode='constant', constant_values=0)

