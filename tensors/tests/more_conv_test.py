#!/usr/bin/env python

import numpy as np
from conv_helpers import *

def corr_backprop(a,b,padval=0):
	(ah,aw)=a.shape
	(bh,bw)=b.shape
	fh=ah-bh+1
	fw=aw-bw+1

	f=np.zeros( (fh,fw) )
	for i in range(0,fh):
		for j in range(0,fw):
			f[i,j]=np.dot( a[i:(i+bh), j:(j+bw)].reshape(-1), b.reshape(-1))
	return f

def corr_gemm_backprop(a,b,padval=0):
	(ah,aw)=a.shape
	(bh,bw)=b.shape
	fh=ah-bh+1
	fw=aw-bw+1

	agemm=np.zeros( (fh*fw, bh*bw) )
	for i in range(0,fh):
		for j in range(0,fw):
			agemm[i*fw+j, :]=a[i:(i+bh), j:(j+bw)].reshape(-1)
	bgemm=b.reshape( (bh*bw, 1) )
	f=np.dot(agemm, bgemm)

	return f.reshape(fh,fw)

def corr_multi_backprop(a,b,padval=0):
	(m,ah,aw)=a.shape
	(m,bh,bw)=b.shape
	fh=ah-bh+1
	fw=aw-bw+1

	f=np.zeros( (fh,fw) )
	for n in range(0,m):
		for i in range(0,fh):
			for j in range(0,fw):
				f[i,j]+=np.dot( a[n,i:(i+bh), j:(j+bw)].reshape(-1), b[n,:,:].reshape(-1))
	return f

def corr_gemm_multi_backprop(a,b,padval=0):
	(m,ah,aw)=a.shape
	(m,bh,bw)=b.shape
	fh=ah-bh+1
	fw=aw-bw+1

	agemm=np.zeros( (fh*fw, m*bh*bw) )
	for i in range(0,fh):
		for j in range(0,fw):
			agemm[i*fw+j, :]=a[:,i:(i+bh), j:(j+bw)].reshape(-1)
	bgemm=b.reshape( (m*bh*bw, 1) )
	f=np.dot(agemm, bgemm)

	return f.reshape(fh,fw)



a=np.random.randint( 10, size=(30,30) )
b=np.random.randint( 10, size=(28,28) )

f=corr_backprop(a,b)
print a.shape
print b.shape
print f.shape

f_gemm=corr_gemm_backprop(a,b)
print a.shape
print b.shape
print f.shape
print (f==f_gemm).all()

a=np.random.randint( 10, size=(128,30,30) )
b=np.random.randint( 10, size=(128,28,28) )

f=corr_multi_backprop(a,b)
print a.shape
print b.shape
print f.shape

f_gemm=corr_gemm_multi_backprop(a,b)
print a.shape
print b.shape
print f.shape
print (f==f_gemm).all()

a=np.random.randint( 10, size=(128,30,30,4) )
b=np.random.randint( 10, size=(128,28,28,5) )

f=corr_tensor_backprop(a,b)
print a.shape
print b.shape
print f.shape

f_gemm=corr_gemm_tensor_backprop(a,b)
print a.shape
print b.shape
print f.shape
print (f==f_gemm).all()

