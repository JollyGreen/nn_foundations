#!/usr/bin/env python

import time
import numpy as np
from pad_helpers import *


N=128
H=14
W=14
C=32

z=np.random.randn(N,H,W,C)

start=time.time()
pad=pad_brute(z,1)
stop=time.time()
brutetime=stop-start

start=time.time()
pad_fast=pad_fast(z,1)
stop=time.time()
fasttime=stop-start

print "times: ", brutetime, fasttime, (pad==pad_fast).all()
