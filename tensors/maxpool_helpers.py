#!/usr/bin/env python

import numpy as np
import time

def maxpool_bruteforce_nchw(z):
	(N,C,H,W)=z.shape
	a=np.zeros( (N,C,H/2,W/2) )
	switches=np.zeros( (N,C,H/2,W/2), dtype=int)
	for n in range(0,N):
		for c in range(0,C):
			for i in range(0,H/2):
				for j in range(0,W/2):
					patch=z[n, c, i*2:(i*2+2), j*2:(j*2+2)].reshape(-1)
					switch=np.argmax( patch )
					switches[n,c,i,j]=switch
					a[n,c,i,j]=patch[switch]
	return [a,switches]

def maxpool_bruteforce_nhwc(z):
	(N,H,W,C)=z.shape
	a=np.zeros( (N,H/2,W/2,C) )
	switches=np.zeros( (N,H/2,W/2,C), dtype=int)
	for n in range(0,N):
		for c in range(0,C):
			for i in range(0,H/2):
				for j in range(0,W/2):
					patch=z[n, i*2:(i*2+2), j*2:(j*2+2), c].reshape(-1)
					switch=np.argmax( patch )
					switches[n,i,j,c]=switch
					a[n,i,j,c]=patch[switch]
	return [a,switches]

def maxpool_back_bruteforce_nchw(z, switches):
	(N,C,OH,OW)=z.shape
	FH=2
	FW=2
	H=OH*2
	W=OW*2
	a=np.zeros( (N,C,H,W) )
	for n in range(0,N):
		for c in range(0,C):
			for i in range(0,OH):
				for j in range(0,OW):
					val=z[n,c,i,j]
					patch=np.zeros(FH*FW)
					patch[switches[n,c,i,j]]=val

					a[n,c,i*2:(i*2+2),j*2:(j*2+2)]=patch.reshape(FH,FW)
	return a

def maxpool_back_bruteforce_nhwc(z, switches):
	(N,OH,OW,C)=z.shape
	FH=2
	FW=2
	H=OH*2
	W=OW*2
	a=np.zeros( (N,H,W,C) )
	cntr=0
	for n in range(0,N):
		for c in range(0,C):
			for i in range(0,OH):
				for j in range(0,OW):
					val=z[n,i,j,c]
					patch=np.zeros(FH*FW)
					patch[switches[n,i,j,c]]=val

					a[n,i*2:(i*2+2),j*2:(j*2+2),c]=patch.reshape(FH,FW)
	return a

def maxpool_indices_nchw(zshape):
	(N,C,H,W)=zshape
	FH=2
	FW=2
	stride=2

	OH=H/2
	OW=W/2

	r=[0,0,1,1]
	r=np.tile(r,OW*OH)
	r0=np.repeat(np.arange(0,OH)*stride, FH*FW*OW)
	r=r+r0
	#r=[
	#	0,0,1,1,0,0,1,1,0,0,1,1,
	#	2,2,3,3,2,2,3,3,2,2,3,3,
	#	4,4,5,5,4,4,5,5,4,4,5,5
	#]
	c=[0,1,0,1]
	c=np.tile(c,OH*OW)
	c0=np.tile(np.repeat(np.arange(0,OW)*stride, FH*FW), OH)
	c=c+c0
	#c=[
	#	0,1,0,1,2,3,2,3,4,5,4,5,
	#	0,1,0,1,2,3,2,3,4,5,4,5,
	#	0,1,0,1,2,3,2,3,4,5,4,5
	#]
	r=np.tile(r,C)
	c=np.tile(c,C)
	k=np.repeat(np.arange(0,C), FH*FW*OH*OW)
	return [k,r,c]

def maxpool_indices_nhwc(zshape):
	(N,H,W,C)=zshape
	FH=2
	FW=2
	stride=2

	OH=H/2
	OW=W/2

	r=[0,0,1,1]
	r=np.tile(r,OW*OH)
	r0=np.repeat(np.arange(0,OH)*stride, FH*FW*OW)
	r=r+r0
	c=[0,1,0,1]
	c=np.tile(c,OH*OW)
	c0=np.tile(np.repeat(np.arange(0,OW)*stride, FH*FW), OH)
	c=c+c0
	r=np.tile(r,C)
	c=np.tile(c,C)
	k=np.repeat(np.arange(0,C), FH*FW*OH*OW)
	return [r,c,k]
	
def maxpool_nchw(z):
	(N,C,H,W)=z.shape
	FH=2
	FW=2
	stride=2
	OH=H/2
	OW=W/2

	[k,r,c]=maxpool_indices_nchw(z.shape)

	cols=z[:,k,r,c].reshape(-1,FH*FW)

	switches=np.argmax(cols, axis=1)
	vals=cols[np.arange(0,cols.shape[0]), switches]
	return [vals.reshape(N,C,OH,OW), switches.reshape( (N,C,OH,OW) )]



def maxpool_nhwc(z):
	(N,H,W,C)=z.shape
	FH=2
	FW=2
	stride=2
	OH=H/2
	OW=W/2

	[r,c,k]=maxpool_indices_nhwc(z.shape)

	cols=z[:,r,c,k]

	cols=cols.reshape(-1,FH*FW)

	switches=np.argmax(cols, axis=1)
	vals=cols[np.arange(0,cols.shape[0]), switches]

	switches=switches.reshape(N,C,OH,OW).transpose(0,2,3,1)
	vals=vals.reshape(N,C,OH,OW).transpose(0,2,3,1)
	return [vals, switches]

def maxpool_back_nchw(a, switches):
	(N,C,OH,OW)=a.shape
	H=OH*2
	W=OW*2
	FH=2
	FW=2
	stride=2

	o=np.zeros( (N,C,H,W) )
	switches=switches.reshape(-1)

	vals=a.reshape(-1)
	fills=np.zeros( (vals.shape[0], FH*FW) )
	fills[np.arange(0, vals.shape[0]), switches]=vals
	
	fills=fills.reshape(N, -1)
	[k,r,c]=maxpool_indices_nchw( (N,C,H,W) )

	np.add.at(o, (slice(None),k,r,c), fills)

	return o

def maxpool_back_nhwc(a, switches):
	(N,OH,OW,C)=a.shape
	H=OH*2
	W=OW*2
	FH=2
	FW=2
	stride=2

	o=np.zeros( (N,H,W,C) )
	switches=switches.transpose(0,3,1,2).reshape(-1)

	vals=a.transpose(0,3,1,2).reshape(-1)
	fills=np.zeros( (vals.shape[0], FH*FW) )
	fills[np.arange(0, vals.shape[0]), switches]=vals
	
	fills=fills.reshape(N, -1)
	[r,c,k]=maxpool_indices_nhwc( (N,H,W,C) )

	np.add.at(o, (slice(None),r,c,k), fills)

	return o	


def fill_nchw(zshape):
	(N,C,H,W)=zshape
	z=np.zeros( (N,C,H,W), dtype=int)
	imgcntr=0
	channelcntr=0
	for n in range(0,N):
		channelcntr=0
		for c in range(0,C):
			pixelcntr=0
			for h in range(0,H):
				for w in range(0,W):
					z[n,c,h,w]=imgcntr+channelcntr+pixelcntr
					pixelcntr=pixelcntr+1
			channelcntr=channelcntr+100
		imgcntr=imgcntr+1000
	return z

def fill_nhwc(zshape):
	(N,H,W,C)=zshape
	z=np.zeros( (N,H,W,C), dtype=int)
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
	return z


