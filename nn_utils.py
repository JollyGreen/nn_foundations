#!/usr/bin/env python

import numpy as np

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



