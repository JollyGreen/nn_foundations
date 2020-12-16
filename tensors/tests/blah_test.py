#!/usr/bin/env python

import numpy as np

def im2col_indices(ashape, FH,FW,stride):
	(N,H,W,C)=ashape
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
	return [newr,newc,k]

def custom_maxpool_nhwc(a, FH, FW, stride):

	[newr,newc,k]=im2col_indices(a.shape, FH,FW,stride)

	rowtest=a[:,newr,newc,k].reshape(-1,4)

	switches=np.argmax(rowtest, axis=1)
	maxs=rowtest[range(0,switches.shape[0]),switches]
	return [maxs.reshape(N,H/2,W/2,C), switches, newr,newc,k]

def max_zeros(z):
	(N,H,W,C)=z.shape
	a=np.zeros( (N,H,W,C) )
	for n in range(0,N):
		for channel in range(0,C):
			for r in range(0,H,2):
				for c in range(0,W,2):
					patch=z[n, r:(r+2), c:(c+2), channel].ravel()
					switch=np.argmax( patch )
					switchr=int(np.floor(switch/2))
					switchc=switch % 2
					a[n,r+switchr,c+switchc,channel]=patch[switch]
	return a

def fast_max_zeros(a, FH, FW, stride):

	[newr,newc,k]=im2col_indices(a.shape, FH,FW,stride)

	print newr
	print newc
	print k
	tmp=a[:,newr,newc,k]
	print tmp.shape
	rowtest=tmp.reshape(-1,4)

	print rowtest.astype(int)
	switches=np.argmax(rowtest, axis=1)
	maxs=rowtest[range(0,switches.shape[0]),switches]
	return [maxs.reshape(N,H/2,W/2,C), switches, newr,newc,k]

N=3
H=6
W=6
C=4

z=np.zeros( (N,H,W,C) )

imgcntr=0
channelcntr=0
for n in range(0,N):
	channelcntr=0
	for c in range(0,C):
		pixelcntr=0
		for h in range(0,H):
			for w in range(0,W):
				z[n,h,w,c]=imgcntr+channelcntr+pixelcntr
				pixelcntr=pixelcntr+1
		channelcntr=channelcntr+100
	imgcntr=imgcntr+1000

z1=z.transpose(0,3,1,2)
z1=z1.reshape(N*C,1,H,W)

a=max_zeros(z)


fast_max_zeros(z,2,2,2)

