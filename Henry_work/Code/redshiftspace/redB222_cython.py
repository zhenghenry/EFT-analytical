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

start_time = time.time()

gm.get_context().precision = 200
gm.get_context().allow_complex = True

wig = False

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

lenfbabis = 33
lenfdiogo = 16


# write all the babis functions
fbabisparamtab = np.zeros((lenfbabis,3),dtype=object)
fbabisparamtab[0]=[ mpc(0), 0, 0]
fbabisparamtab[1]=[ -kpeak2 - kuv2*1j, 1, 1]
fbabisparamtab[2]=[ -kpeak2 + kuv2*1j, 1, 1]
fbabisparamtab[3]=[ -kpeak3 - 1j*kuv3, 0, 1]
fbabisparamtab[4]=[ -kpeak3 + 1j*kuv3, 0, 1]
fbabisparamtab[5]=[ -kpeak3 - 1j*kuv3, 0, 2]
fbabisparamtab[6]=[ -kpeak3 + 1j*kuv3, 0, 2]
fbabisparamtab[7]=[ -kpeak4 - 1j*kuv4, 0, 1]
fbabisparamtab[8]=[ -kpeak4 + 1j*kuv4, 0, 1]
fbabisparamtab[9]=[ -kpeak1 - 1j*kuv1, 0, 1]
fbabisparamtab[10]=[ -kpeak1 + 1j*kuv1, 0, 1]
fbabisparamtab[11]=[ -kpeak2 - 1j*kuv2, 1, 2]
fbabisparamtab[12]=[ -kpeak2 + 1j*kuv2, 1, 2]
fbabisparamtab[13]=[ -kpeak3 - 1j*kuv3, 0, 3]
fbabisparamtab[14]=[ -kpeak3 + 1j*kuv3, 0, 3]
fbabisparamtab[15]=[ -kpeak4 - 1j*kuv4, 0, 2]
fbabisparamtab[16]=[ -kpeak4 + 1j*kuv4, 0, 2]
fbabisparamtab[17]=[ -kpeak1 - 1j*kuv1, 0, 2]
fbabisparamtab[18]=[ -kpeak1 + 1j*kuv1, 0, 2]
fbabisparamtab[19]=[ -kpeak2 - 1j*kuv2, 1, 3]
fbabisparamtab[20]=[ -kpeak2 + 1j*kuv2, 1, 3]
fbabisparamtab[21]=[ -kpeak3 - 1j*kuv3, 0, 4]
fbabisparamtab[22]=[ -kpeak3 + 1j*kuv3, 0, 4]
fbabisparamtab[23]=[ -kpeak4 - 1j*kuv4, 0, 3]
fbabisparamtab[24]=[ -kpeak4 + 1j*kuv4, 0, 3]
fbabisparamtab[25]=[ -kpeak1 - 1j*kuv1, 0, 3]
fbabisparamtab[26]=[ -kpeak1 + 1j*kuv1, 0, 3]
fbabisparamtab[27]=[ -kpeak2 - 1j*kuv2, 1, 4]
fbabisparamtab[28]=[ -kpeak2 + 1j*kuv2, 1, 4]
fbabisparamtab[29]=[ -kpeak3 - 1j*kuv3, 0, 5]
fbabisparamtab[30]=[ -kpeak3 + 1j*kuv3, 0, 5]
fbabisparamtab[31]=[ -kpeak4 - 1j*kuv4, 0, 4]
fbabisparamtab[32]=[ -kpeak4 + 1j*kuv4, 0, 4]


# load coefficients
ctabfolder = 'simple_redshiftspace_bispectrum/B222redctabks/'

outputfolder = 'simple_redshiftspace_bispectrum/B222redjmatbabisnowig_cython/'

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]
print(filelist)

def compute_jmat(filename):

	ctab_load = np.loadtxt(ctabfolder + filename, dtype = object)
	ctab = np.zeros((len(ctab_load),4), dtype = object)

	for i in range(len(ctab)):
		ctab[i, 0] = round(float(ctab_load[i, 0]))
		ctab[i, 1] = round(float(ctab_load[i, 1]))
		ctab[i, 2] = round(float(ctab_load[i, 2]))
		# ctab[i, 3] = mpc(str(ctab_load[i, 3]))
		ctab[i, 3] = complex(str(ctab_load[i, 3]))

	k1 = mpfr(str.split(filename,'_')[1])/10
	k2 = mpfr(str.split(filename,'_')[2])/10
	k3 = mpfr(str.split(str.split(filename,'_')[3],'.csv')[0])/10

	k12 = k1**2
	k22 = k2**2
	k32 = k3**2

	print(k12,k22,k32)

	ltriantable = np.zeros((len(fbabisparamtab),len(fbabisparamtab),len(fbabisparamtab)),dtype=object)
	for i1 in reversed(range(lenfbabis)):
		for i2 in reversed(range(lenfbabis)):
			for i3 in reversed(range(lenfbabis)):
				# if ltriantable[i1,i2,i3] == 0:
				# print(i1,i2,i3)
				res = 0+0j
				for i in range(len(ctab)):
					if ctab[i,3] != 0:
						# print(ctab)
						term = ctab[i,3] * Ltrian(-ctab[i, 1] + fbabisparamtab[i1, 1], fbabisparamtab[i1, 2], -ctab[i, 0] + fbabisparamtab[i2, 1],
												 fbabisparamtab[i2, 2], -ctab[i, 2] + fbabisparamtab[i3, 1], fbabisparamtab[i3, 2], k12,
												 k22, k32, fbabisparamtab[i1, 0], fbabisparamtab[i2, 0], fbabisparamtab[i3, 0])
						# print('i',i,'term = ', "{0:.10f}".format(term))			
						# print(-ctab[i, 1] + fbabisparamtab[i1, 1], fbabisparamtab[i1, 2], -ctab[i, 0] + fbabisparamtab[i2, 1],
						# 						 fbabisparamtab[i2, 2], -ctab[i, 2] + fbabisparamtab[i3, 1], fbabisparamtab[i3, 2],term)
						res += term
				
				ltriantable[i1,i2,i3] = res

				fill_with_conjugate(i1,i2,i3,res,ltriantable)
	out_arr = np.column_stack((np.repeat(np.arange(lenfbabis),lenfbabis),ltriantable.reshape(lenfbabis**2,-	1)))
	out_df = pd.DataFrame(out_arr)
	out_df.to_csv(outputfolder+'jmatoutk_'+ str(float(10*k1)) + '_' + str(float(10*k2)) + '_' + str(float(10*k3)) + '_' +'.csv',index = False)
	print("--- %s seconds ---" % (time.time() - start_time))
# def computeker(i1,i2,i3):
# 	res = 0+0j
# 	for i in range(len(ctab)):
# 		if ctab[i,3] != 0:
# 			term = ctab[i,3] * Ltrian(ctab[i, 1] + fbabisparamtab[i1, 1], fbabisparamtab[i1, 2], ctab[i, 0] + fbabisparamtab[i2, 1],
# 									 fbabisparamtab[i2, 2], ctab[i, 2] + fbabisparamtab[i3, 1], fbabisparamtab[i3, 2], k12,
# 									 k22, k32, fbabisparamtab[i1, 0], fbabisparamtab[i2, 0], fbabisparamtab[i3, 0])
# 			# print('i',i,'term = ', "{0:.10f}".format(term))			
# 			res += term
# 	# print(type(res))
# 	ltriantable[i1,i2,i3] = res
# 	return res

def fill_with_conjugate(i1,i2,i3,res,ltriantable):
	# fills table with corresponding conjugate value after we calculate one value
	
	resconj = res.conjugate()

	if i1 > 0 and i2 > 0 and i3 > 0:
		if i1%2 == 0 and i2%2 == 0 and i3%2 == 0:
			if ltriantable[i1-1,i2-1,i3-1] == 0:
				ltriantable[i1-1,i2-1,i3-1] = resconj
		elif i1%2 == 0 and i2%2 == 0 and i3%2 == 1:
			if ltriantable[i1-1,i2-1,i3+1] == 0:
				ltriantable[i1-1,i2-1,i3+1] = resconj
		elif i1%2 == 0 and i2%2 == 1 and i3%2 == 0:
			if ltriantable[i1-1,i2+1,i3-1] == 0:
				ltriantable[i1-1,i2+1,i3-1] = resconj
		elif i1%2 == 0 and i2%2 == 1 and i3%2 == 1:
			if ltriantable[i1-1,i2+1,i3+1] == 0:
				ltriantable[i1-1,i2+1,i3+1] = resconj
		elif i1%2 == 1 and i2%2 == 0 and i3%2 == 0:
			if ltriantable[i1+1,i2-1,i3-1] == 0:
				ltriantable[i1+1,i2-1,i3-1] = resconj
		elif i1%2 == 1 and i2%2 == 0 and i3%2 == 1:
			if ltriantable[i1+1,i2-1,i3+1] == 0:
				ltriantable[i1+1,i2-1,i3+1] = resconj
		elif i1%2 == 1 and i2%2 == 1 and i3%2 == 1:
			if ltriantable[i1+1,i2+1,i3+1] == 0:
				ltriantable[i1+1,i2+1,i3+1] = resconj

	elif i1 == 0 and i2 > 0 and i3 > 0:
		if i2%2 == 0 and i3%2 == 0:
			if ltriantable[0,i2-1,i3-1] == 0:
				ltriantable[0,i2-1,i3-1] = resconj
		elif i2%2 == 0 and i3%2 == 1:
			if ltriantable[0,i2-1,i3+1] == 0:
				ltriantable[0,i2-1,i3+1] = resconj
		elif i2%2 == 1 and i3%2 == 0:
			if ltriantable[0,i2+1,i3-1] == 0:
				ltriantable[0,i2+1,i3-1] = resconj
		elif i2%2 == 1 and i3%2 == 1:
			if ltriantable[0,i2+1,i3+1] == 0:
				ltriantable[0,i2+1,i3+1] = resconj

	elif i1 > 0 and i2 == 0 and i3 > 0:
		if i1%2 == 0 and i3%2 == 0:
			if ltriantable[i1-1,0,i3-1] == 0:
				ltriantable[i1-1,0,i3-1] = resconj
		elif i1%2 == 0 and i3%2 == 1:
			if ltriantable[i1-1,0,i3+1] == 0:
				ltriantable[i1-1,0,i3+1] = resconj
		elif i1%2 == 1 and i3%2 == 0:
			if ltriantable[i1+1,0,i3-1] == 0:
				ltriantable[i1+1,0,i3-1] = resconj
		elif i1%2 == 1 and i3%2 == 1:
			if ltriantable[i1+1,0,i3+1] == 0:
				ltriantable[i1+1,0,i3+1] = resconj

	elif i1 > 0 and i2 > 0 and i3 == 0:
		if i1%2 == 0 and i2%2 == 0:
			if ltriantable[i1-1,i2-1,0] == 0:
				ltriantable[i1-1,i2-1,0] = resconj
		elif i1%2 == 0 and i2%2 == 1:
			if ltriantable[i1-1,i2+1,0] == 0:
				ltriantable[i1-1,i2+1,0] = resconj
		elif i1%2 == 1 and i2%2 == 0:
			if ltriantable[i1+1,i2-1,0] == 0:
				ltriantable[i1+1,i2-1,0] = resconj
		elif i1%2 == 1 and i2%2 == 1:
			if ltriantable[i1+1,i2+1,0] == 0:
				ltriantable[i1+1,i2+1,0] = resconj
	return 0


# print(computeker(0,1,7))
# print(computeker(0,1,8))
# print(ltriantable[0,1,8])



# routine to fill up J table
# for i1 in reversed(range(lenfbabis)):
# 	for i2 in reversed(range(lenfbabis)):
# 		print(i1,i2)
# 		for i3 in reversed(range(lenfbabis)):
# 			if ltriantable[i1,i2,i3] == 0:
# 				# print(i1,i2,i3)
# 				res = computeker(i1,i2,i3)

# 				fill_with_conjugate(i1,i2,i3,res)

# Output the table to csv 
# out_arr = np.column_stack((np.repeat(np.arange(lenfbabis),lenfbabis),ltriantable.reshape(lenfbabis**2,-1)))
# out_df = pd.DataFrame(out_arr)
# out_df.to_csv(outputfolder+'jmatoutk_'+ str(float(10*k1)) + '_' + str(float(10*k2)) + '_' + str(float(10*k3)) + '_newmasses' +'.csv',index = False)
# print("--- %s seconds ---" % (time.time() - start_time))

# print(gm.get_context().precision)
# print(filelist[12])
# compute_jmat(filelist[12])
for file in filelist:
	print(file)
	compute_jmat(file)
	start_time = time.time()
	print("--- %s seconds ---" % (time.time() - start_time))


'''

# Load previously calculated table
print(outputfolder+'jmatoutk_'+str(float(10*k1)) + '_' + str(float(10*k2)) + '_' + str(float(10*k3)) + '_newmasses' +'.csv')
load_ltriantable = np.loadtxt(outputfolder+'jmatoutk_'+str(float(10*k1)) + '_' + str(float(10*k2)) + '_' + str(float(10*k3)) + '_newmasses' +'.csv', dtype = complex, delimiter=',', skiprows = 1, usecols = range(1,lenfbabis+1))
print("shape of load_ltriantable: ", load_ltriantable.shape)

ltriantable_new = load_ltriantable.reshape(lenfbabis,lenfbabis,lenfbabis)

print("shape of ltriantable_new: ", ltriantable_new.shape)

# upload change of basis matrix
matarray = np.fromfile('matdiogotobabis', np.complex128)
matdiogotobabis = np.reshape(matarray,(16,33))
print("shape of matdiogotobabis: ", matdiogotobabis.shape)


# perform the matrix multiplication
ltriantablefin = np.einsum('ijk,li,mj,nk', ltriantable_new, matdiogotobabis, matdiogotobabis, matdiogotobabis)
ltriantablefin = np.real_if_close(ltriantablefin, tol = 10)
lentriantable = np.shape(ltriantablefin)[0]
print(np.shape(ltriantablefin), lentriantable, lenfdiogo)
# print(ltriantablefin[1])
# output final B222 table to csv
out_arr = np.column_stack((np.repeat(np.arange(lenfdiogo),lenfdiogo),ltriantablefin.reshape(lenfdiogo**2,-1)))
out_df = pd.DataFrame(out_arr)
out_df.to_csv(outputfolder+'jmatoutk_fdiogo_'+str(float(10*k1)) + '_' + str(float(10*k2)) + '_' + str(float(10*k3)) +'.csv',index = False)
print("--- %s seconds ---" % (time.time() - start_time))

'''
