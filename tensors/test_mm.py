#!/usr/bin/env python

import time
import numpy as np

from scipy.linalg import get_blas_funcs
rX=25088
cX=288

rY=cX
cY=64

X=np.random.randn(rX,cX).astype(float)
Y=np.random.randn(rY,cY).astype(float)

gemm = get_blas_funcs("gemm", [X, Y])

start=time.time()
a=gemm(1, X, Y)
stop=time.time()
gemm_time=stop-start

start=time.time()
b=np.dot(X,Y)
stop=time.time()
dot_time=stop-start

print gemm_time, dot_time, np.allclose(a,b)


rX=288
cX=25088

rY=64
cY=rX

X=np.random.randn(rX,cX).astype(float)
Y=np.random.randn(rY,cY).astype(float)

gemm = get_blas_funcs("gemm", [X, Y])

start=time.time()
a=gemm(1, Y, X)
stop=time.time()
gemm_time=stop-start

start=time.time()
b=np.dot(Y,X)
stop=time.time()
dot_time=stop-start

print gemm_time, dot_time, np.allclose(a,b)
