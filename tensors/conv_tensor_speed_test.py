#!/usr/bin/env python

import time
import numpy as np
from conv_helpers import *

print "4D Tensor"
f=np.random.randint( 10, size=(3,3,3,10) )
a=np.random.randint( 10, size=(128,28,28,3) )

start=time.time()
o_hat_4d_tensor=corr4d_tensor(a,f,padval=1)
end=time.time()
print end-start
print o_hat_4d_tensor.shape

print "4D GEMM Tensor"
start=time.time()
o_hat_4d_gemm_tensor=corr4d_gemm_tensor(a,f,padval=1)
end=time.time()
print end-start
print o_hat_4d_gemm_tensor.shape
print (o_hat_4d_tensor==o_hat_4d_gemm_tensor).all()
