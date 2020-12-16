#!/usr/bin/env python

import time
import numpy as np
from maxpool_helpers import fill_nhwc
from maxpool_helpers import fill_nchw
from pad_helpers import *

def extract_cols_brute(z, fshape):
	(N,H,W,C)=z.shape
	(FH,FW)=fshape

	OH=H-FH+1
	OW=W-FW+1

	o=np.zeros( (N*OH*OW, FH*FW*C) )

	for n in range(0,N):
		for r in range(0,OH):
			for c in range(0,OW):
				for channel in range(0,C):
					o[n*OH*OW+r*OW+c, channel*FH*FW:(channel+1)*FH*FW]=z[n, r:(r+FH), c:(c+FW), channel].reshape(-1)
	return o

def extract_cols_t(z, fshape):
	(N,H,W,C)=z.shape
	(FH,FW)=fshape

	OH=H-FH+1
	OW=W-FW+1

	o=np.zeros( (N*OH*OW, FH*FW*C) )

	z=z.transpose( (0,3,1,2) )
	for n in range(0,N):
		for r in range(0,OH):
			for c in range(0,OW):
				o[n*OH*OW+r*OW+c, :]=z[n, :, r:(r+FH), c:(c+FW)].reshape(-1)
	return o

def extract_cols(z, fshape):
	(N,H,W,C)=z.shape
	(FH,FW)=fshape

	OH=H-FH+1
	OW=W-FW+1

	FLEN=FH*FW*C

	r=[0,0,0,1,1,1,2,2,2]
	c=[0,1,2,0,1,2,0,1,2]
	r=np.tile(r,C)
	c=np.tile(c,C)
	k=np.repeat(range(0,C), FH*FW)

	r=np.tile(r, OW)

	c0=np.repeat(range(0,OW), FH*FW*C)
	c=np.tile(c, OW)
	c=c+c0

	k=np.tile(k, OW)

	r0=np.repeat(range(0,OH), FH*FW*C*OW)
	r=np.tile(r, OH)
	r=r+r0

	c=np.tile(c, OH)

	k=np.tile(k, OH)

	#print r.reshape(-1, FLEN)
	#print c.reshape(-1, FLEN)
	#print k.reshape(-1, FLEN)

	cols=z[:,r,c,k].reshape(-1, FLEN)
	#print cols.shape
	#print cols[0,:]
	#print cols[1,:]
	return cols

def im2col_indices_331_nchw(zshape):
	(N,C,H,W)=zshape
	FH=3
	FW=3
	stride=1

	OH=H-FH+1
	OW=W-FW+1

	r=[0,0,0,1,1,1,2,2,2]
	r=np.tile(np.tile(r,C), OH*OW)
	r0=np.repeat(np.arange(0,OH)*stride, FH*FW*C*OW)
	r=r+r0
	c=[0,1,2,0,1,2,0,1,2]
	c=np.tile(np.tile(c,C), OH*OW)
	c0=np.tile(np.repeat(np.arange(0,OW)*stride, FH*FW*C), OH)
	c=c+c0
	k=np.repeat(np.arange(0,C), FH*FW)
	k=np.tile(k,OH*OW)
	return [k,r,c]



def rows_brute_nhwc(z, f):
	(N,H,W,C)=z.shape
	(fh,fw,ic,oc)=f.shape

	padz=pad_fast(z,1)
	(N,PADH,PADW,C)=padz.shape

	OH=PADH-fh+1
	OW=PADW-fw+1

	flen=fh*fw*ic	
	tmp=np.zeros( (N*OH*OW, flen) )
	for i in range(0,N):
		for r in range(0,OH):
			for c in range(0,OW):
				tmp[i*OH*OW+r*OW+c,:]=padz[i,r:(r+fh),c:(c+fw),:].reshape(-1)
				#tmp[i*oh*ow+r*ow+c,:]=pada[i,:,r:(r+fh),c:(c+fw)].reshape(-1)
	return tmp

def rows_brute_nchw(z, f):
	(N,C,H,W)=z.shape
	(FH,FW,ic,oc)=f.shape

	OH=H-FH+1
	OW=W-FW+1

	flen=FH*FW*C
	tmp=np.zeros( (N*OH*OW, flen), dtype=int)
	for i in range(0,N):
		for r in range(0,OH):
			for c in range(0,OW):
				tmp[i*OH*OW+r*OW+c,:]=z[i,:,r:(r+FH),c:(c+FW)].reshape(-1)
				#tmp[i*oh*ow+r*ow+c,:]=pada[i,:,r:(r+fh),c:(c+fw)].reshape(-1)
	return tmp

def cols_brute(z, f):
	(N,H,W,C)=z.shape
	(fh,fw,ic,oc)=f.shape

	padz=pad_fast(z,1)
	(N,PADH,PADW,C)=padz.shape

	OH=PADH-fh+1
	OW=PADW-fw+1

	flen=fh*fw*ic	
	tmp=np.zeros( (flen, N*OH*OW) )
	for i in range(0,N):
		for r in range(0,OH):
			for c in range(0,OW):
				tmp[:,i*OH*OW+r*OW+c]=padz[i,r:(r+fh),c:(c+fw),:].reshape(-1)
				#tmp[i*oh*ow+r*ow+c,:]=pada[i,:,r:(r+fh),c:(c+fw)].reshape(-1)
	return tmp

def im2cols_fast(z,f):
	(N,C,H,W)=z.shape
	(FH,FW,ic,oc)=f.shape

	OH=H-FH+1
	OW=W-FW+1

	flen=FH*FW*ic	
	[k,r,c]=im2col_indices_331_nchw(z.shape)
	#print r
	#print c
	#print k
	rows=z[:,k,r,c]
	print rows.shape
	print rows
	rows=rows.reshape(-1,flen)
	return rows

	
N=3
H=4
W=4
C=2

z=fill_nhwc( (N,H,W,C) ).astype(int)
z2=fill_nchw( (N,C,H,W) ).astype(int)
f=np.random.randn(3,3,C,32)
#print z.transpose(0,3,1,2)

start=time.time()
o=extract_cols_brute(z, (3,3))
stop=time.time()
brutetime=stop-start

start=time.time()
o2=extract_cols(z, (3,3))
stop=time.time()
fasttime=stop-start

start=time.time()
o3=extract_cols_t(z, (3,3))
stop=time.time()
ttime=stop-start

print "Brute", brutetime
print "Transpose", ttime, brutetime/ttime
print "Fast Index", fasttime, brutetime/fasttime
print "Match", (o==o2).all(), (o==o3).all()

rows=rows_brute_nhwc(z, f)
cols=cols_brute(z, f)
print (rows==cols.transpose()).all()

start=time.time()
rows=rows_brute_nchw(z2,f)
stop=time.time()
brutetime=stop-start


start=time.time()
rows_fast=im2cols_fast(z2,f)
stop=time.time()
fasttime=stop-start

print brutetime,fasttime
print (rows==rows_fast).all()
print rows.transpose()
