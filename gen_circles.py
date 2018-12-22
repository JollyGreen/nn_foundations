#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

m=1000
scale=0.25

offsetx=0.0
offsety=0.0

vals=1.0*np.array(range(0,m))*(1.0/float(m))
x1=offsetx+np.cos(2*np.pi*vals)+np.random.rand(m)*scale
y1=offsety+np.sin(2*np.pi*vals)+np.random.rand(m)*scale
c0=np.zeros(m)

x2=offsetx+0.75*np.cos(2*np.pi*vals)+np.random.rand(m)*scale
y2=offsety+0.75*np.sin(2*np.pi*vals)+np.random.rand(m)*scale
c1=np.ones(m)

xvals=np.concatenate([x1,x2])
yvals=np.concatenate([y1,y2])
cvals=np.concatenate([c0,c1])

arr=range(0,len(xvals))
np.random.shuffle(arr)

xvals=xvals[arr]
yvals=yvals[arr]
cvals=cvals[arr]

plt.plot(x1,y1, '+')
plt.plot(x2,y2, '+')
plt.show()


f=file('circles.txt', 'w')
for i in range(0,len(xvals)):
	print >> f, '%.3f, %.3f, %1d' % ( xvals[i], yvals[i], cvals[i])
f.close()
