#!/usr/bin/env python

import time
import numpy as np

from maxpool_helpers import *

def timing_custom_maxpool_back_nchw(a, switches):
	(N,C,OH,OW)=a.shape
	H=OH*2
	W=OW*2
	FH=2
	FW=2
	stride=2

	start=time.time()
	o=np.zeros( (N,C,H,W) )
	stop=time.time()
	print "Init", stop-start

	switches=switches.reshape(-1)

	vals=a.reshape(-1)

	start=time.time()
	fills=np.zeros( (vals.shape[0], FH*FW) )
	fills[np.arange(0, vals.shape[0]), switches]=vals
	fills=fills.reshape(N, -1)
	stop=time.time()
	print fills.shape
	print "Fills", stop-start

	
	start=time.time()
	[k,r,c]=maxpool_indices_nchw( (N,C,H,W) )
	stop=time.time()
	print "indices", stop-start


	print k.shape
	print r.shape
	print c.shape
	start=time.time()
	np.add.at(o, (slice(None),k,r,c), fills)
	stop=time.time()
	print "addto", stop-start

	return o
def custom_maxpool_back_nchw(a, switches):
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

def speedup_maxpool_back_nchw(a, switches):
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

	allgoodidxs=fills!=0.0
	idxs=np.arange(0,fills.shape[1])
	for n in range(0,N):
		goodidxs=idxs[allgoodidxs[n,:]]

		goodfills=fills[n,goodidxs]
		goodk=k[goodidxs]
		goodr=r[goodidxs]
		goodc=c[goodidxs]
		np.add.at(o[n,:,:,:], (goodk,goodr,goodc), goodfills)

	return o

def speedup_maxpool_back_nhwc(a, switches):
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

	allgoodidxs=fills!=0.0
	idxs=np.arange(0,fills.shape[1])
	for n in range(0,N):
		goodidxs=idxs[allgoodidxs[n,:]]

		goodfills=fills[n,goodidxs]
		goodk=k[goodidxs]
		goodr=r[goodidxs]
		goodc=c[goodidxs]
		np.add.at(o[n,:,:,:], (goodr,goodc,goodk), goodfills)

	return o	


N=128
C=32
H=28
W=28

#z=fill_nchw( (N,C,H,W) )
z=np.random.randn(N,C,H,W)
z2=np.random.randn(N,H,W,C)

start=time.time()
[a,aswitches]=maxpool_nchw(z)
stop=time.time()
forwardtime=stop-start

start=time.time()
out=custom_maxpool_back_nchw(a,aswitches)
stop=time.time()
backwardtime=stop-start

print "Forward", forwardtime, "Backward", backwardtime

start=time.time()
speedout=speedup_maxpool_back_nchw(a, aswitches)
stop=time.time()
speedtime=stop-start

print speedtime, (out==speedout).all()
[b,bswitches]=maxpool_nhwc(z2)
start=time.time()
speedoutb=speedup_maxpool_back_nhwc(b, bswitches)
stop=time.time()
speedoutc=maxpool_back_nhwc(b, bswitches)
print stop-start, (speedoutb==speedoutc).all()
