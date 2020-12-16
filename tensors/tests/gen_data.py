#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

m=500
scale=0.95

x1=np.random.uniform(size=m)*scale
y1=np.random.uniform(size=m)*scale
c0=np.zeros(m)

x2=np.random.uniform(size=m)*scale+1.0
y2=np.random.uniform(size=m)*scale+1.0
c1=np.ones(m)

xvals=np.concatenate([x1,x2])
yvals=np.concatenate([y1,y2])
cvals=np.concatenate([c0,c1])

arr=range(0,len(xvals))
np.random.shuffle(arr)

xvals=xvals[arr]
yvals=yvals[arr]
cvals=cvals[arr]

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()

print x1.shape
print xvals.shape

f=file('out.txt', 'w')
for i in range(0,len(xvals)):
	print >> f, '%.3f, %.3f, %1d' % ( xvals[i], yvals[i], cvals[i])
f.close()
