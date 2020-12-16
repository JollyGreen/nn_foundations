#!/usr/bin/env python

import numpy as np

def custom_maxpool(a, FH, FW, stride):
	(N,C,H,W)=a.shape
	OH=H/2
	OW=W/2

	#a=np.arange(0,N*H*W*C).reshape( (N,C,H,W) )
	r0=np.repeat(np.arange(0,W,2),FH*FW*OW)
	r=np.tile([0,0,1,1], OW*OH)

	c0=np.tile(stride*np.repeat(np.arange(0,OW), FH*FW), OH)
	c=np.tile([0,1,0,1], OW*OH)

	rows=a[:,:,r+r0,c+c0].reshape(-1,4)

	maxs=np.max(rows, axis=1)
	return maxs.reshape(N,C,H/2,W/2)

def custom_maxpool_nhwc(a, FH, FW, stride):
	(N,H,W,C)=a.shape
	OH=H/2
	OW=W/2

	#a=np.arange(0,N*H*W*C).reshape( (N,C,H,W) )
	r0=np.repeat(np.arange(0,W,2),FH*FW*OW)
	r=np.tile([0,0,1,1], OW*OH)

	c0=np.tile(stride*np.repeat(np.arange(0,OW), FH*FW), OH)
	c=np.tile([0,1,0,1], OW*OH)

	newr=np.tile(r+r0,C)
	newc=np.tile(c+c0,C)
	k=np.repeat(np.arange(0,C), FH*FW*OH*OW)

	print r

	print c

	print k

	rowstest=a[:,newr,newc,k].reshape(-1,4)
	rows=a[:,r+r0,c+c0,:].transpose(0,2,1).reshape(-1,4)


	print (rows==rowstest).all()
	maxs=np.max(rows, axis=1)
	return maxs.reshape(N,H/2,W/2,C)

N=3
H=6
W=6
C=3

FH=2
FW=2
stride=2
cntr=0

a=np.zeros( (N,C,H,W), dtype=int)
for n in range(0,N):
	for c in range(0,C):
		for h in range(0,H):
			for w in range(0,W):
				a[n,c,h,w]=cntr
				cntr=cntr+1 

print a[0,0,:,:]

custom_maxpool(a, 2, 2, 2)

cntr=0
a=np.zeros( (N,H,W,C), dtype=int)
for n in range(0,N):
	for c in range(0,C):
		for h in range(0,H):
			for w in range(0,W):
				a[n,h,w,c]=cntr
				cntr=cntr+1 

print a[0,:,:,0]

maxs=custom_maxpool_nhwc(a, 2, 2, 2)
print maxs
