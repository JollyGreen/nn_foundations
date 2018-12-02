#!/usr/bin/env python

import numpy as np
import struct

def parse_file(filename):
	f=file(filename, 'r')
	xvals=[]
	yvals=[]
	cvals=[]
	for line in f:
		vals=line.split(',')
		xval=float(vals[0].strip())
		yval=float(vals[1].strip())
		cval=int(vals[2].strip())

		xvals.append(xval)
		yvals.append(yval)
		cvals.append(cval)
	f.close()
	points=np.zeros( (2,len(xvals)) )
	classes=np.zeros( (2,len(xvals)) )

	for i in range(0,len(xvals)):
		points[0][i]=xvals[i]
		points[1][i]=yvals[i]
		classes[0][i]=float(cvals[i])
		classes[1][i]=1.0-float(cvals[i])
	return [points, classes]


def parse_mnist_train(filename_digits, filename_labels, numdigits):
	digits=np.array([])
	onehot=np.array([])
	try:
		fdigits=open(filename_digits, "rb")
		magicnumber=struct.unpack('>i', fdigits.read(4))[0]
		numimages=struct.unpack('>i', fdigits.read(4))[0]
		numrows=struct.unpack('>i', fdigits.read(4))[0]
		numcols=struct.unpack('>i', fdigits.read(4))[0]
		print 'magicnumber:',magicnumber,'numimages:',numimages,'numrows:',numrows,'numcols:',numcols
		digits=np.transpose(np.fromfile(fdigits, dtype='>B', count=numrows*numcols*numdigits).reshape(numdigits, numrows*numcols))
	finally:
		fdigits.close()

	try:
		flabels=open(filename_labels, "rb")
		magicnumber=struct.unpack('>i', flabels.read(4))[0]
		numlabels=struct.unpack('>i', flabels.read(4))[0]
		print 'magicnumber:',magicnumber,'numlabels:',numlabels
		labels=np.array(np.fromfile(flabels, dtype='>B', count=numdigits)).reshape(-1)
		onehot=np.zeros((10, numdigits))
		onehot[labels, np.arange(numdigits)]=1
	finally:
		flabels.close()

	return [digits, onehot]
	
