#!/usr/bin/env python

import numpy as np
from maxpool_helpers import *

N=128
C=32
H=28
W=28

z=fill_nchw( (N,C,H,W) )
z2=fill_nhwc( (N,H,W,C) )


start=time.time()
[a, aswitches]=maxpool_nchw(z)
maxsb=maxpool_back_nchw(a, aswitches)
stop=time.time()
fasttime=stop-start

start=time.time()
[b, bswitches]=maxpool_bruteforce_nchw(z)
maxsa=maxpool_back_bruteforce_nchw(a, aswitches)
stop=time.time()
brutetime_nchw=stop-start

print "Maxpool matches", (a==b).all()
print "Maxpool backward matches", (maxsa==maxsb).all()
print brutetime_nchw, fasttime, brutetime_nchw/fasttime


start=time.time()
[c, cswitches]=maxpool_bruteforce_nhwc(z2)
maxsc=maxpool_back_bruteforce_nhwc(c, cswitches)
stop=time.time()
brutetime_nhwc=stop-start

start=time.time()
[d,dswitches]=maxpool_nhwc(z2)
maxsd=maxpool_back_nhwc(d, dswitches)
stop=time.time()
fasttime_nhwc=stop-start
print (c==d).all()
print (maxsc==maxsd).all()
print brutetime_nhwc, fasttime_nhwc, brutetime_nhwc/fasttime_nhwc

def maxpool_batch_nchw(z):
	(N,C,H,W)=z.shape
	FH=2
	FW=2
	stride=2
	OH=H/2
	OW=W/2
	
	tmp=z.reshape(N*C,1,H,W)
	[k,r,c]=maxpool_indices_nchw(tmp.shape)

	cols=tmp[:,:,r,c]
	cols=cols.reshape(-1,FH*FW)
	switches=np.argmax(cols, axis=1)
	vals=cols[np.arange(0,cols.shape[0]), switches]
	return [vals.reshape(N,C,OH,OW), switches.reshape( (N,C,OH,OW) )]

def maxpool_back_batch_nchw(a, switches):
	(N,C,OH,OW)=a.shape
	H=OH*2
	W=OW*2
	FH=2
	FW=2
	stride=2

	o=np.zeros( (N*C,1,H,W) )
	switches=switches.reshape(-1)

	vals=a.reshape(-1)
	fills=np.zeros( (vals.shape[0], FH*FW) )
	fills[np.arange(0, vals.shape[0]), switches]=vals
	
	fills=fills.reshape(N*C, -1)
	[k,r,c]=maxpool_indices_nchw( (N*C,1,H,W) )
	np.add.at(o, (slice(None),k,r,c), fills)

	return o.reshape(N,C,H,W)

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

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
	""" An implementation of im2col based on some fancy indexing """
	# Zero-pad the input
	p = padding
	x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

	k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

	cols = x_padded[:, k, i, j]
	C = x.shape[1]
	cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
	return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
	""" An implementation of col2im based on fancy indexing and np.add.at """
	N, C, H, W = x_shape
	H_padded, W_padded = H + 2 * padding, W + 2 * padding
	x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
	k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
	cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
	cols_reshaped = cols_reshaped.transpose(2, 0, 1)
	np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
	if padding == 0:
		return x_padded
	return x_padded[:, :, padding:-padding, padding:-padding]


def maxpool_forward_nhwc(X, size=2, stride=2):
	(N,C,H,W)=X.shape
	h_out=H/2
	w_out=W/2

	# Let say our input X is 5x10x28x28
	# Our pooling parameter are: size = 2x2, stride = 2, padding = 0
	# i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

	# First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
	X_reshaped = X.reshape(N * C, 1, H, W)

	# The result will be 4x9800
	# Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
	X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
	# Next, at each possible patch location, i.e. at each column, we're taking the max index
	max_idx = np.argmax(X_col, axis=0)

	# Finally, we get all the max value at each column
	# The result will be 1x9800
	out = X_col[max_idx, range(max_idx.size)]

	# Reshape to the output size: 14x14x5x10
	out = out.reshape(h_out, w_out, N, C)

	# Transpose to get 5x10x14x14 output
	out = out.transpose(2, 3, 0, 1)

	return [out, max_idx]

def maxpool_backward_nhwc(dout,max_idx):
	(N,C,OH,OW)=dout.shape
	Xshape=(N,C,OH*2,OW*2)
	size=2
	stride=2

	# X_col and max_idx are the intermediate variables from the forward propagation step

	# Suppose our output from forward propagation step is 5x10x14x14
	# We want to upscale that back to 5x10x28x28, as in the forward step

	# 4x9800, as in the forward step
	#dX_col = np.zeros_like(X_col)
	dX_col = np.zeros( (4, max_idx.shape[0]) )

	# 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
	# Transpose step is necessary to get the correct arrangement
	dout_flat = dout.transpose(2, 3, 0, 1).ravel()

	# Fill the maximum index of each column with the gradient

	# Essentially putting each of the 9800 grads
	# to one of the 4 row in 9800 locations, one at each column
	dX_col[max_idx, range(max_idx.size)] = dout_flat

	# We now have the stretched matrix of 4x9800, then undo it with col2im operation
	# dX would be 50x1x28x28
	dX = col2im_indices(dX_col, (N * C, 1, H, W), size, size, padding=0, stride=stride)

	# Reshape back to match the input dimension: 5x10x28x28
	dX = dX.reshape(Xshape)
	return dX

start=time.time()
[e,eswitches]=maxpool_batch_nchw(z)
maxse=maxpool_back_batch_nchw(e, eswitches)
stop=time.time()
fasttime_batch=stop-start
print (e==b).all()
print (maxse==maxsb).all()
print brutetime_nchw, fasttime_batch, brutetime_nchw/fasttime_batch

print "ONLINE IM2COL"

start=time.time()
[custom, customswitches]=maxpool_nchw(z)
maxscustom=maxpool_back_nchw(custom, customswitches)
stop=time.time()
customtime=stop-start

start=time.time()
[online, switches]=maxpool_forward_nhwc(z)
maxsonline=maxpool_backward_nhwc(online, switches)
stop=time.time()
onlinetime=stop-start
print customtime, fasttime_nhwc, onlinetime, customtime/onlinetime, fasttime_nhwc/onlinetime, (maxscustom==maxsonline).all()
