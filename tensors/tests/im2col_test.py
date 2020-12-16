#!/usr/bin/env python

import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
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

	print i0
	print i0.reshape(-1,1)
	print i1
	print i1.reshape(1,-1)

	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

	return (k, i, j)


#x=np.random.randint( 10, size=(1,6,6,1) )
N=1
H=6
W=6
C=1

x=np.zeros( (N,H,W,C) )

cntr=0
for n in range(0,N):
	for c in range(0,C):
		for h in range(0,H):
			for w in range (0,W):
				x[n,h,w,c]=cntr
				cntr=cntr+1

k,i,j=get_im2col_indices(x.shape,2,2,0,2)

print k
print i
print j

cols=x[:,i,j,k]
print cols.shape

C=x.shape[3]
cols=cols.reshape(2*2*C,-1)

cols=cols.transpose(1,0).reshape(-1,4)

print x[0,:,:,0]
print x[0,:,:,1]
print cols
