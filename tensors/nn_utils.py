#!/usr/bin/env python

import numpy as np
import struct

def parse_file(filename):
	f=file(filename, 'r')
	xvals=[]
	yvals=[]
	cvals=[]
	m=0
	for line in f:
		vals=line.split(',')
		xval=float(vals[0].strip())
		yval=float(vals[1].strip())
		cval=int(vals[2].strip())

		xvals.append(xval)
		yvals.append(yval)
		cvals.append(cval)
	f.close()
	m=len(xvals)
	points=np.zeros( (m,2) )
	classes=np.zeros( (m,2) )

	for i in range(0,len(xvals)):
		points[i][0]=xvals[i]
		points[i][1]=yvals[i]
		classes[i][0]=float(cvals[i])
		classes[i][1]=1.0-float(cvals[i])
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
		digits=np.fromfile(fdigits, dtype='>B', count=numrows*numcols*numdigits).reshape(numdigits, numrows*numcols)
	finally:
		fdigits.close()

	try:
		flabels=open(filename_labels, "rb")
		magicnumber=struct.unpack('>i', flabels.read(4))[0]
		numlabels=struct.unpack('>i', flabels.read(4))[0]
		print 'magicnumber:',magicnumber,'numlabels:',numlabels
		labels=np.array(np.fromfile(flabels, dtype='>B', count=numdigits)).reshape(-1)
		onehot=np.zeros((numdigits,10))
		onehot[np.arange(numdigits),labels]=1
	finally:
		flabels.close()

	return [digits, onehot]
	
