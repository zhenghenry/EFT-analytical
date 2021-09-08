import numpy as np 
import mpmath as mp
import os
import pandas as pd
import time
from functools import lru_cache
import babispython_v15 as b
from babispython_v15 import GLOBAL_PREC

start_time = time.time()

mp.mp.dps = GLOBAL_PREC

wig = False
k = 1
kpeak1 = -0.034
kpeak2 = -0.001
kpeak3 = -0.000076
kpeak4 = -0.0000156
kuv1 = 0.069
kuv2 = 0.0082
kuv3 = 0.0013
kuv4 = 0.0000135
# kpeak1 = mp.mpf(str(kpeak1))
# kpeak2 = mp.mpf(str(kpeak2))
# kpeak3 = mp.mpf(str(kpeak3))
# kpeak4 = mp.mpf(str(kpeak4))
# kuv1 = mp.mpf(str(kuv1))
# kuv2 = mp.mpf(str(kuv2))
# kuv3 = mp.mpf(str(kuv3))
# kuv4 = mp.mpf(str(kuv4))

mass1 = -kpeak1 - 1j*kuv1
mass1conj = mp.conj(mass1)
mass2 = -kpeak2 - 1j*kuv2
mass2conj = mp.conj(mass2)
mass3 = -kpeak3 - 1j*kuv3
mass3conj = mp.conj(mass3)
mass4 = -kpeak4 - 1j*kuv4
mass4conj = mp.conj(mass4)


kminBW = mp.mpf(str(0.09))
kmaxBW = mp.mpf(str(0.242))
kminBWhigh = mp.mpf(str(0.21))
kmaxBWhigh = mp.mpf(str(0.5))
imaxwig = 7
imaxhigh = 4

m0 = []
delBW = []
m0high = []
delBWhigh = []
for i in range(1,imaxwig+1):
	m0.append(74/100*kminBW*mp.exp(mp.log(kmaxBW/kminBW)*11/10*i/imaxwig))
	delBW.append(6/100*mp.exp(mp.log(kmaxBW/kminBW)*88/100*i/imaxwig))
for i in range(1,imaxhigh+1):
	m0high.append(108/100*kminBWhigh*mp.exp(mp.log(kmaxBWhigh/kminBWhigh)*2/10*i/imaxhigh))
	delBWhigh.append(17/1000*mp.exp(i/imaxhigh))


ctabfolder = 'B411ctabks/'
if wig:
	outputfolder = 'B411jmatbabis'
else:
	outputfolder = 'B411jmatbabisnowig_cython/'

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]
# print(filelist)


if wig:
	lenfbabis = 85
else:
	lenfbabis = 33
#def compute_jmat(filename):
filename = 'B411ctab_0.05_0.05_0.05_.csv'
# filename = filelist[0]


ctab_load = np.loadtxt(ctabfolder + filename, dtype = object)
ctab = ctab_load.astype(float) 

k1 = mp.mpf(str.split(filename,'_')[1])/10
k2 = mp.mpf(str.split(filename,'_')[2])/10
k3 = mp.mpf(str.split(filename,'_')[3])/10
print(k1,k2,k3)
k1sq = k1**2
k2sq = k2**2
k3sq = k3**2
fbabisparamtab = np.zeros((lenfbabis,4),dtype=object)
fbabisparamtab[0]=[k, 0, 0, 0]
fbabisparamtab[1]=[k, -kpeak2 - kuv2*1j, 1, 1]
fbabisparamtab[2]=[k, -kpeak2 + kuv2*1j, 1, 1]
fbabisparamtab[3]=[k, -kpeak3 - 1j*kuv3, 0, 1]
fbabisparamtab[4]=[k, -kpeak3 + 1j*kuv3, 0, 1]
fbabisparamtab[5]=[k, -kpeak3 - 1j*kuv3, 0, 2]
fbabisparamtab[6]=[k, -kpeak3 + 1j*kuv3, 0, 2]
fbabisparamtab[7]=[k, -kpeak4 - 1j*kuv4, 0, 1]
fbabisparamtab[8]=[k, -kpeak4 + 1j*kuv4, 0, 1]
fbabisparamtab[9]=[k, -kpeak1 - 1j*kuv1, 0, 1]
fbabisparamtab[10]=[k, -kpeak1 + 1j*kuv1, 0, 1]
fbabisparamtab[11]=[k, -kpeak2 - 1j*kuv2, 1, 2]
fbabisparamtab[12]=[k, -kpeak2 + 1j*kuv2, 1, 2]
fbabisparamtab[13]=[k, -kpeak3 - 1j*kuv3, 0, 3]
fbabisparamtab[14]=[k, -kpeak3 + 1j*kuv3, 0, 3]
fbabisparamtab[15]=[k, -kpeak4 - 1j*kuv4, 0, 2]
fbabisparamtab[16]=[k, -kpeak4 + 1j*kuv4, 0, 2]
fbabisparamtab[17]=[k, -kpeak1 - 1j*kuv1, 0, 2]
fbabisparamtab[18]=[k, -kpeak1 + 1j*kuv1, 0, 2]
fbabisparamtab[19]=[k, -kpeak2 - 1j*kuv2, 1, 3]
fbabisparamtab[20]=[k, -kpeak2 + 1j*kuv2, 1, 3]
fbabisparamtab[21]=[k, -kpeak3 - 1j*kuv3, 0, 4]
fbabisparamtab[22]=[k, -kpeak3 + 1j*kuv3, 0, 4]
fbabisparamtab[23]=[k, -kpeak4 - 1j*kuv4, 0, 3]
fbabisparamtab[24]=[k, -kpeak4 + 1j*kuv4, 0, 3]
fbabisparamtab[25]=[k, -kpeak1 - 1j*kuv1, 0, 3]
fbabisparamtab[26]=[k, -kpeak1 + 1j*kuv1, 0, 3]
fbabisparamtab[27]=[k, -kpeak2 - 1j*kuv2, 1, 4]
fbabisparamtab[28]=[k, -kpeak2 + 1j*kuv2, 1, 4]
fbabisparamtab[29]=[k, -kpeak3 - 1j*kuv3, 0, 5]
fbabisparamtab[30]=[k, -kpeak3 + 1j*kuv3, 0, 5]
fbabisparamtab[31]=[k, -kpeak4 - 1j*kuv4, 0, 4]
fbabisparamtab[32]=[k, -kpeak4 + 1j*kuv4, 0, 4]

# fbabisparamtab[0]=[k, 0, 0, 0]
# fbabisparamtab[1]=[k, -kpeak2 - kuv2*1j, 1, 2]
# fbabisparamtab[1]=[k, mass3, 2, 6]
# fbabisparamtab[2]=[k, mass3conj, 2, 6]

if wig:
	for i in range(imaxwig):
		fbabisparamtab[41 + 4*i] = [k, -m0[i]**2 - 1j*m0[i]*delBW[i], 1, 1]
		fbabisparamtab[41 + 4*i+1] = [k, -m0[i]**2 + 1j*m0[i]*delBW[i], 1, 1]
		fbabisparamtab[41 + 4*i+2] = [k, -m0[i]**2 - 1j*m0[i]*delBW[i], 1, 2]
		fbabisparamtab[41 + 4*i+3] = [k, -m0[i]**2 + 1j*m0[i]*delBW[i], 1, 2]

	for i in range(imaxhigh):
		fbabisparamtab[41 + 4*i+4*imaxwig] = [k, -m0high[i]**2 - 1j*delBWhigh[i], 1, 1]
		fbabisparamtab[41 + 4*i+4*imaxwig+1] = [k, -m0high[i]**2 + 1j*delBWhigh[i], 1, 1]
		fbabisparamtab[41 + 4*i+4*imaxwig+2] = [k, -m0high[i]**2 - 1j*delBWhigh[i], 1, 2]
		fbabisparamtab[41 + 4*i+4*imaxwig+3] = [k, -m0high[i]**2 + 1j*delBWhigh[i], 1, 2]


ltriantable = np.zeros((len(fbabisparamtab)),dtype = object)
print(GLOBAL_PREC)



def computeker(i1):
	res = 0
	for i in range(len(ctab)):
		coef = ctab[i,-1]

		if coef == 0:
			term = 0
		else:
			if ctab[i, 1] != 0:
				term = ctab[i,6]*b.Ltrian(ctab[i, 2], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
										 fbabisparamtab[i1, 3], ctab[i, 4], 0, mp.mpmathify(k1**2),
										 mp.mpmathify(k2**2), mp.mpmathify(k3**2), 0, fbabisparamtab[i1, 1], 0)
			elif ctab[i, 3] != 0:
				term = ctab[i,6]*b.Ltrian(ctab[i, 2], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
										 fbabisparamtab[i1, 3], ctab[i, 4], 0, mp.mpmathify(k1**2),
										 mp.mpmathify(k3**2), mp.mpmathify(k2**2), 0, fbabisparamtab[i1, 1], 0)
			elif ctab[i, 5] != 0:
				term = ctab[i,6]*b.Ltrian(ctab[i, 2], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
										 fbabisparamtab[i1, 3], ctab[i, 4], 0, mp.mpmathify(k3**2),
										 mp.mpmathify(k2**2), mp.mpmathify(k1**2), 0, fbabisparamtab[i1, 1], 0)	
			# if ctab[i,-2] == 123:
			# 	term = coef*b.Ltrian(ctab[i, 1], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
			# 							 fbabisparamtab[i1, 3], ctab[i, 2], 0, k1sq,
			# 							 k2sq, k3sq, 0, fbabisparamtab[i1, 1], 0)	
			# elif ctab[i,-2] == 132:				
			# 	term = coef*b.Ltrian(ctab[i, 1], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
			# 							 fbabisparamtab[i1, 3], ctab[i, 2], 0, k1sq,
			# 							 k3sq, k2sq, 0, fbabisparamtab[i1, 1], 0)	
			# elif ctab[i,-2] == 213:				
			# 	term = coef*b.Ltrian(ctab[i, 1], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
			# 							 fbabisparamtab[i1, 3], ctab[i, 2], 0, k2sq,
			# 							 k1sq, k3sq, 0, fbabisparamtab[i1, 1], 0)	
			# elif ctab[i,-2] == 231:				
			# 	term = coef*b.Ltrian(ctab[i, 1], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
			# 							 fbabisparamtab[i1, 3], ctab[i, 2], 0, k2sq,
			# 							 k3sq, k1sq, 0, fbabisparamtab[i1, 1], 0)	
			# elif ctab[i,-2] == 321:				
			# 	term = coef*b.Ltrian(ctab[i, 1], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
			# 							 fbabisparamtab[i1, 3], ctab[i, 2], 0, k3sq,
			# 							 k2sq, k1sq, 0, fbabisparamtab[i1, 1], 0)	
			# elif ctab[i,-2] == 312:				
			# 	term = coef*b.Ltrian(ctab[i, 1], 0, ctab[i, 0] + fbabisparamtab[i1, 2],
			# 							 fbabisparamtab[i1, 3], ctab[i, 2], 0, k3sq,
			# 							 k1sq, k2sq, 0, fbabisparamtab[i1, 1], 0)	
			res += term
	
	ltriantable[i1] = mp.chop(res, tol = 10**(-GLOBAL_PREC+10))
	return res



for i1 in range(len(fbabisparamtab)):
	computeker(i1)


# print(computeker(34))
#mp.nprint(ltriantable[1])

pd.set_option("precision", GLOBAL_PREC)
np.set_printoptions(precision=GLOBAL_PREC)
out_df = pd.DataFrame(ltriantable, dtype = object)
out_df.to_csv(outputfolder+'jmatoutk_'+str(10*k1)+'_' + str(10*k2) + '_' + str(10*k3) + '_' +'.csv',index = False)
print("--- %s seconds ---" % (time.time() - start_time))




# for file in filelist:
# 	compute_jmat(file)
# 	print("--- %s seconds ---" % (time.time() - start_time))