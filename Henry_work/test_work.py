import numpy as np 

import gmpy2 as gm
from gmpy2 import mpfr, mpc

import os
import pandas as pd
import time
from functools import lru_cache
# from babiscython_v1 import Ltrian
# from babiscython_v2_testspeed import Ltrian
from babiscython_v4_ubuntu import Ltrian
from Jfunc_cython_v4 import computeJ
import mpmath as mp
start_time = time.time()
import babispython_v15 as b

gm.get_context().precision = 200
gm.get_context().allow_complex = True
# from test_jfunc_v2 import computeJ
# print(GLOBAL_PREC)
mp.mp.dps = 80

#mp.nprint((b.TriaN(2,0,7,1,1,1,mp.mpf('0.00001'),mp.mpf('0.00001'),mp.mpf('0.00001'))),20)

kpeak1 = -0.034
kpeak2 = -0.001
kpeak3 = -0.000076
kpeak4 = -0.0000156
kuv1 = 0.069
kuv2 = 0.0082
kuv3 = 0.0013
kuv4 = 0.0000135
kpeak1 = mpfr(str(kpeak1))
kpeak2 = mpfr(str(kpeak2))
kpeak3 = mpfr(str(kpeak3))
kpeak4 = mpfr(str(kpeak4))
kuv1 = mpfr(str(kuv1))
kuv2 = mpfr(str(kuv2))
kuv3 = mpfr(str(kuv3))
kuv4 = mpfr(str(kuv4))

mass1 = -kpeak1 - 1j*kuv1
# mass1conj = mp.conj(mass1)
mass2 = -kpeak2 - 1j*kuv2
# mass2conj = mp.conj(mass2)
mass3 = -kpeak3 - 1j*kuv3
# mass3conj = mp.conj(mass3)
mass4 = -kpeak4 - 1j*kuv4
# mass4conj = mp.conj(mass4)

k1 = mpfr('0.01')
k2 = mpfr('0.01')
k3 = mpfr('0.01')

k12 = mp.mpmathify('0.01')
k22 = mp.mpmathify('0.01')
k32 = mp.mpmathify('0.01')


n = 6




# numk2 = k1**2
# denk2 = k2**2
# ksum2 = k3**2

# coef1, coef2, coef3 = b.num_four_pow(1,5,denk2,0,mass1)
# term = coef1*numk2**2 + numk2*b.k1dotk2(numk2,denk2,ksum2)**2*coef2/denk2 + coef3*b.k1dotk2(numk2,denk2,ksum2)**4/denk2**2
# mp.nprint(tri_dim(4,1,5,numk2,denk2,ksum2,0,mass1), 10)
# mp.nprint(TriaMaster(k1**2,k2**2,k3**2,mass1,mass2,mass3))
# mp.nprint(("Here:",computeJ(-4,1,0,0,0,4,k1,k2,k3)))
mp.nprint(Ltrian(-2,4,-2,4,0,4,k1**2,k2**2,k3**2,mass3,mass3,mass3))
mp.nprint(computeJ(1,1,1,1,1,1,mpfr(0.05),mpfr(0.05),mpfr(0.05)))
# mp.nprint(b.Ltrian(-2,4,-2,4,0,4,0.01,0.01,0.01,-0.000076+ 0.0013j,-0.000076+ 0.0013j,-0.000076+ 0.0013j))
# mp.nprint(('here:', TriaMaster(k1**2,k2**2,k3**2,mass1,mass2,mass3)),6)