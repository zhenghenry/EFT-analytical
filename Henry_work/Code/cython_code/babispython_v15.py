import numpy as np
from functools import lru_cache
import math
import mpmath as mp
from mpmath import *
from numba import njit, complex128

GLOBAL_PREC = 50
CHOP_TOL = 1e-40
mp.dps = GLOBAL_PREC
mp.pretty = True


@njit(fastmath=True)
def gammaC(z):

	q0 = 75122.6331530 
	q1 = 80916.6278952 
	q2 = 36308.2951477 
	q3 = 8687.24529705
	q4 = 1168.92649479 
	q5 = 83.8676043424 
	q6 = 2.50662827511
	
	if z >= 0:
		p1 = (q0 + q1*z + q2*z**2 + q3*z**3 + q4*z**4 + q5*z**5 + q6*z**6) / (z*(z+1)*(z+2)*(z+3)*(z+4)*(z+5)*(z+6))
		res = p1 * (z + 11/2)**(z+1/2) * np.exp(-z-11/2)
	else:
		p1 = (q0 + q1*(1-z) + q2*(1-z)**2 + q3*(1-z)**3 + q4*(1-z)**4 + q5*(1-z)**5 + q6*(1-z)**6)/((1-z)*(2-z)*(3-z)*(4-z)*(5-z)*(6-z)*(7-z))
		p2 = p1*(1 - z + 11/2)**(1-z+1/2)*np.exp(-1+z-11/2)
		res = np.pi/np.sin(np.pi*z)/p2
	return res

# Utility function to calculate trinomial coefficient
def trinomial(n, k, i):
	# coefficient in (x + y + z)^n corresponding to x^k y^i z^(n-k-i)
	return binomial(n, k + i) * binomial(k + i, i)

#The following 7 functions allow us to calculate dim reg results 
# when we have powers in the numerator
def get_coef(n1,k2exp,q2exp,kmq = True):
	# get coefficient of the expansion of (k-q)^2n1 
	# that has the form (k^2)^k2exp * (q^2)^q2exp * (k.q)^(n1-k2exp-q2exp)
	kqexp = n1 - k2exp - q2exp
	if kmq:
		prefac = power((-2),kqexp)
	else:
		prefac = power(2,kqexp)
	# return prefac*factorial(n1)/factorial(k2exp)/factorial(q2exp)/factorial(kqexp)
	return prefac * trinomial(n1, k2exp, q2exp)

def num_terms(n1,kmq = True):
	# expands terms of the type (k-q)^(2n1) if kmq = True
	# expands terms of the type (k+q)^(2n1) if kmq = False

	term_list = []
	exp_list = []

	for k2exp in arange(nint(n1+1)):
		for q2exp in arange(nint(n1+1-k2exp)):
			term_list.append(get_coef(n1,k2exp,q2exp,kmq))
			exp_list.append([k2exp,q2exp,n1-k2exp-q2exp])

	return term_list, exp_list

def dim_gen(expnum,expden,m):
	if m == 0:
		return 0
	return (2/sqrt(pi()))*gamma(expnum + 3/2)*gamma(expden - expnum - 3/2)/gamma(expden)*m**(expnum - expden + 3/2)


def get_coef_simple(n1,kmq2exp):
	# get coefficient of the expansion of ((k-q)^2+m)^n1 
	# that has the form ((k-q)^2)^kmq2exp * m^(n1-kmq2exp)
	# (aka just binomial expansion)
	return binomial(n1, kmq2exp)
	# mexp = n1 - kmq2exp
	# return factorial(n1)/factorial(kmq2exp)/factorial(mexp)

def expand_massive_num(n1):
	#expand term of the form ((kmq)^2+mNum)^expnum
	#returns:
	# - term_list: list of coefficients
	# - exp_list: list of exponents [expkmq2,mNumexp]

	term_list = []
	exp_list = []

	for kmq2exp in arange(n1+1):
		term_list.append(get_coef_simple(n1,kmq2exp))
		exp_list.append([kmq2exp,n1-kmq2exp])

	return term_list, exp_list

@lru_cache(None)
def dim_result(expkmq2,expden,k2,mDen,kmq = True):
	# computes integrals of the type ((k-q)^2)^expnum / (q^2 + mDen)^expden
	# if kmq is TRUE, consider (k-q)
	# if kmq is FALSE, consider (k+q)

	term_list, exp_list = num_terms(expkmq2,kmq)
	res_list = 0

	for i in range(len(exp_list)):

		k2exp = exp_list[i][0]
		q2exp = exp_list[i][1]
		kqexp = exp_list[i][2]

		if kqexp%2 == 0:
			if kqexp != 0:
				res_list += 1/(1+kqexp)*term_list[i]*dim_gen(q2exp+kqexp/2,expden,mDen)*(k2)**(k2exp+kqexp/2)
			
			else:
				res_list += term_list[i]*dim_gen(q2exp,expden,mDen)*(k2)**(k2exp)
	return res_list

@lru_cache(None)
def compute_massive_num(expnum,expden,k2,mNum,mDen,kmq = True):
	# computes integrals of the type ((k-q)^2+mNum)^expnum / (q^2 + mDen)^expden
	# by summing over many dim_results
	
	if mNum == 0:
		return dim_result(expnum,expden,k2,mDen)

	term_list, exp_list = expand_massive_num(expnum)
	res_list = 0

	for i in range(len(exp_list)):
		kmq2exp = exp_list[i][0]
		mNumexp = exp_list[i][1]

		res_list += term_list[i]*(mNum**mNumexp)*dim_result(kmq2exp,expden,k2,mDen)

	return res_list


#Function that calculates Tadpoles
@lru_cache(None)
def TadN(n, m):
	if m == 0:
		return 0
	if n == 0:
		return 0
	if n == 1:
		return -2 * sqrt(pi()) * sqrt(m)
	if n < 0:
		return 0

	return dim_gen(0,n,m)

@lru_cache(None)
def BubN(n1, n2, k2, m1, m2):

	if n1 == 0:
		# print('n2', n2, 'm2', m2, 'TadN', TadN(n2, m2))
		return TadN(n2, m2)

	if n2 == 0:
		# print('n1', n1, 'm1', m1, 'TadN',TadN(n1, m1))
		return TadN(n1, m1)

	if n1 == 1 and n2 == 1:
		# print('BubMaster',BubMaster(k2, m1, m2))
		return BubMaster(k2, m1, m2)

	k1s = k2 + m1 + m2
	jac = k1s**2 - 4*m1*m2

	dim = 3
	if n1 > 1:
		nu1 = n1 - 1
		nu2 = n2
		Ndim = dim - nu1 - nu2

		cpm0 = k1s
		cmp0 = -2*m2/nu1*nu2
		c000 = (2*m2-k1s)/nu1*Ndim - 2*m2 + k1s*nu2/nu1
	elif n2 > 1:
		nu1 = n1
		nu2 = n2-1
		Ndim = dim - nu1 - nu2

		cpm0 = -2*m1/nu2*nu1
		cmp0 = k1s
		c000 = (2*m1 - k1s)/nu2*Ndim + k1s/nu2*nu1 - 2*m1
	
	#code to deal with numerators
	if n1 < 0 or n2 < 0:
		if m1 == 0 and m2 == 0:
			return 0
		if n1 < 0 and n2 > 0:
			# m1 is the mass in the numerator
			# m2 is the mass in the denominator 
			return compute_massive_num(-n1,n2,k2,m1,m2)
		elif n2 < 0 and n1 > 0:
			# m2 is the mass in the numerator
			# m1 is the mass in the denominator
			return compute_massive_num(-n2,n1,k2,m2,m1)
		else:
			print('not confused anymore')
			return 0
			

	c000 = c000/jac
	cmp0 = cmp0/jac
	cpm0 = cpm0/jac

	return c000*BubN(nu1,nu2,k2,m1,m2) + cpm0*BubN(nu1 + 1, nu2 - 1, k2, m1, m2) + cmp0*BubN(nu1 - 1, nu2 + 1, k2, m1, m2)
 

@lru_cache(None)
def TrianKinem(k21, k22, k23, m1, m2, m3):
	
	k1s = k21 + m1 + m2
	k2s = k22 + m2 + m3
	k3s = k23 + m3 + m1

	jac = -4*m1*m2*m3 + k1s**2*m3 + k2s**2*m1 + k3s**2*m2 - k1s*k2s*k3s
	jac = 2*jac

	ks11 = (-4*m1*m2 + k1s**2)/jac
	ks12 = (-2*k3s*m2 + k1s*k2s)/jac
	ks22 = (-4*m2*m3 + k2s**2)/jac
	ks23 = (-2*k1s*m3 + k2s*k3s)/jac
	ks31 = (-2*k2s*m1+k1s*k3s)/jac
	ks33 = (-4*m1*m3+k3s**2)/jac

	kinems = matrix([jac,ks11,ks22,ks33,ks12,ks23,ks31])

	return kinems

@lru_cache(None)
def TriaN(n1, n2, n3, k21, k22, k23, m1, m2, m3):
	# print("n1, n2, n3", n1, n2, n3)
	# n1 = mpmathify(str(n1))
	# n2 = mpmathify(str(n2))
	# n3 = mpmathify(str(n3))

	# print(n1,d1,n2,d2,n3,d3,m1,m2,m3)

	if n1 == 0:
		# print("n1=0, n2=", n2, "n3 = ", n3, BubN(n2, n3, k22, m2, m3))
		return BubN(n2, n3, k22, m2, m3)
	if n2 == 0:
		# print("n2=0, n1=", n1, "n3 = ", n3, BubN(n3, n1, k23, m3, m1))
		return BubN(n3, n1, k23, m3, m1)
	if n3 == 0:
		# print("n3 =0, n1=", n1, "n2 = ", n2, BubN(n1, n2, k21, m1, m2))
		return BubN(n1, n2, k21, m1, m2)


	if n1 == 1 and n2 == 1 and n3 == 1:
		# print("masses", m1, m2, m3)
		if m1 == 0 and m2 == 0 and m3 == 0:
			return TriaMasterZeroMasses(k21,k22,k23)
		# print("TriaMaster = ",TriaMaster(k21, k22, k23, m1, m2, m3))
		return TriaMaster(k21, k22, k23, m1, m2, m3)

	kinem = TrianKinem(k21, k22, k23, m1, m2, m3)
	#jac = kinem[0]
	ks11 = kinem[1]
	ks22 = kinem[2]
	ks33 = kinem[3]
	ks12 = kinem[4]
	ks23 = kinem[5]
	ks31 = kinem[6]

	dim = 3

	if n1 > 1:
		nu1 = n1 - 1
		nu2 = n2
		nu3 = n3

		Ndim = dim - nu1 - nu2 - nu3

		cpm0 = -ks23
		cmp0 = (ks22*nu2)/nu1
		cm0p = (ks22*nu3)/nu1
		cp0m = -ks12
		c0pm = -(ks12*nu2)/nu1
		c0mp = -(ks23*nu3)/nu1
		c000 = (-nu3+Ndim)*ks12/nu1 - (-nu1+Ndim)*ks22/nu1 + (-nu2+Ndim)*ks23/nu1

	elif n2 > 1:
		nu1 = n1
		nu2 = n2 - 1 
		nu3 = n3

		Ndim = dim - nu1 - nu2 - nu3

		cpm0 = (ks33*nu1)/nu2
		cmp0 = -ks23
		cm0p = -(ks23*nu3)/nu2
		cp0m = -(ks31*nu1)/nu2
		c0pm = -ks31
		c0mp = (ks33*nu3)/nu2
		c000 = (-nu1 + Ndim)*ks23/nu2 + (-nu3 + Ndim)*ks31/nu2 - (-nu2 + Ndim)*ks33/nu2

	elif n3 > 1:
		nu1 = n1
		nu2 = n2
		nu3 = n3 - 1 

		Ndim = dim - nu1 - nu2 - nu3


		cpm0 = -(ks31*nu1)/nu3
		cmp0 = -(ks12*nu2)/nu3
		cm0p = -ks12
		cp0m = (ks11*nu1)/nu3
		c0pm = (ks11*nu2)/nu3
		c0mp = -ks31
		c000 = -(-nu3 + Ndim)*ks11/nu3 + (-nu1 + Ndim)*ks12/nu3 + (-nu2 + Ndim)*ks31/nu3
	
	if n1 < 0 or n2 < 0 or n3 < 0:
		# print("tri dim:",n1,n2,n3)
		if n1 < -4 or n2 < -4 or n3 < -4:
			print('ERROR: case not considered -  n1, n2, n3', n1,n2,n3)
		if n1 < 0:
			if n2 > 0 and n3 > 0:
				return tri_dim(-n1,n2,n3,k21,k22,k23,m2,m3)
			elif n2 < 0: 
				return tri_dim_two(-n2,-n1,n3,k22,k23,k21,m3)
			else:
				return tri_dim_two(-n1,-n3,n2,k21,k22,k23,m2)
		if n2 < 0:
			if n1 > 0 and n3 > 0:
				return tri_dim(-n2,n1,n3,k21,k23,k22,m1,m3)
			if n3 < 0:
				return tri_dim_two(-n3,-n2,n1,k23,k21,k22,m1)
		if n3 < 0:
			if n1 > 0 and n2 > 0:
				#print('tri dim',tri_dim(-n3,n1,n2,k23,k21,k22,m1,m2))
				return tri_dim(-n3,n1,n2,k23,k21,k22,m1,m2)	
			print('ERROR: case not considered')
	# print("coefs", c000, c0mp, c0pm, cp0m, cmp0, cpm0)	
	# print(n1, n2, n3, m1,m2,m3,c000*TriaN(nu1, nu2, nu3, k21,k22,k23,m1,m2,m3) + c0mp*TriaN(nu1, nu2-1, nu3+1, k21,k22,k23,m1,m2,m3) +c0pm*TriaN(nu1, nu2+1, nu3-1, k21,k22,k23,m1,m2,m3)+cm0p*TriaN(nu1-1, nu2, nu3+1, k21,k22,k23,m1,m2,m3)+cp0m*TriaN(nu1+1, nu2, nu3-1, k21,k22,k23,m1,m2,m3)+cmp0*TriaN(nu1-1, nu2+1, nu3, k21,k22,k23,m1,m2,m3)+cpm0*TriaN(nu1+1, nu2-1, nu3, k21,k22,k23,m1,m2,m3))
	return  c000*TriaN(nu1, nu2, nu3, k21,k22,k23,m1,m2,m3) + c0mp*TriaN(nu1, nu2-1, nu3+1, k21,k22,k23,m1,m2,m3) +c0pm*TriaN(nu1, nu2+1, nu3-1, k21,k22,k23,m1,m2,m3)+cm0p*TriaN(nu1-1, nu2, nu3+1, k21,k22,k23,m1,m2,m3)+cp0m*TriaN(nu1+1, nu2, nu3-1, k21,k22,k23,m1,m2,m3)+cmp0*TriaN(nu1-1, nu2+1, nu3, k21,k22,k23,m1,m2,m3)+cpm0*TriaN(nu1+1, nu2-1, nu3, k21,k22,k23,m1,m2,m3)

@lru_cache(None)
def tri_dim(n1,d1,d2,numk2,denk2,ksum2,m1,m2):
	# integral of (numk-q)^2n1/(q^2+m1)^2d1/((denk+q)^2+m2)^2d2
	#numerator (numk-q)^2n1 is massless
	#m1 is mass of d1 propagator, which is (q^2+m1)^2d1,
	#m2 is mass of d2 propagator, which is ((denk+q)^2+m2)^2d2
	term_list, exp_list = num_terms(n1,True)
	# print('term list: ', term_list, '\n', 'exp list: ', exp_list, '\n', 'term length: ', len(term_list))
	res_list = 0
	term = 0
	
	for i in range(len(exp_list)):
		k2exp = exp_list[i][0]
		q2exp = exp_list[i][1]
		kqexp = exp_list[i][2]
		if kqexp == 0:
			# in this case our numerator is just (q2)^q2exp
			if q2exp == 0:
				term = BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 1:
				term = BubN(d1-1,d2,denk2,m1,m2)-m1*BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 2:
				term = BubN(d1-2,d2,denk2,m1,m2) -2*m1*BubN(d1-1,d2,denk2,m1,m2) + m1**2*BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 3:
				term = BubN(d1-3,d2,denk2,m1,m2) - 3*m1*BubN(d1-2,d2,denk2,m1,m2) + 3*m1**2*BubN(d1-1,d2,denk2,m1,m2) - m1**3*BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 4:
				term = BubN(d1-4,d2,denk2,m1,m2) - 4*m1*BubN(d1-3,d2,denk2,m1,m2) + 6*m1**2*BubN(d1-2,d2,denk2,m1,m2) - 4*m1**3*BubN(d1-1,d2,denk2,m1,m2) + m1**4*BubN(d1,d2,denk2,m1,m2)
			else:
				print('exceeded calculable power')
			# print('term after first if', term)

		elif kqexp == 1:
			if q2exp == 0:
				term = num_one_pow(d1,d2,denk2,m1,m2)*k1dotk2(numk2,denk2,ksum2)
			elif q2exp == 1:
				term = (num_one_pow(d1-1,d2,denk2,m1,m2)-m1*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
			elif q2exp == 2:
				term = (num_one_pow(d1-2,d2,denk2,m1,m2) - 2*m1*num_one_pow(d1-1,d2,denk2,m1,m2) + m1**2*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
			elif q2exp == 3:
				term = (num_one_pow(d1-3,d2,denk2,m1,m2) - 3*m1*num_one_pow(d1-2,d2,denk2,m1,m2) + 3*m1**2*num_one_pow(d1-1,d2,denk2,m1,m2) - m1**3*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
			else:
				print('exceeded calculable power')
							
			# print('term after second if', term)
		elif kqexp == 2:
			delta_coef, dkcoef = num_two_pow(d1,d2,denk2,m1,m2)
			if q2exp == 0:
				term = (numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/denk2*dkcoef)
			elif q2exp == 1:
				delta_coef2, dkcoef2 = num_two_pow(d1-1,d2,denk2,m1,m2)
				term = -m1*(numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef)
				term += (numk2*delta_coef2 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef2)
			elif q2exp == 2:
				delta_coef2, dkcoef2 = num_two_pow(d1-1,d2,denk2,m1,m2)
				delta_coef3, dkcoef3 = num_two_pow(d1-2,d2,denk2,m1,m2)
				term = (numk2*delta_coef3 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef3)
				term += -2*m1*(numk2*delta_coef2 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef2)
				term += m1**2*(numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/denk2*dkcoef)
			else:
				print('exceeded calculable power')
		
			# print('term after third if', term)
		
		elif kqexp == 3:
			delta_coef, dkcoef = num_three_pow(d1,d2,denk2,m1,m2)
			if q2exp == 0:				
				term = (numk2*delta_coef*k1dotk2(numk2,denk2,ksum2)/(mp.sqrt(denk2)) + dkcoef*k1dotk2(numk2,denk2,ksum2)**3/(denk2*mp.sqrt(denk2)))
			elif q2exp == 1:
				delta_coef2, dkcoef2 = num_three_pow(d1-1,d2,denk2,m1,m2)
				term = (numk2*delta_coef2*k1dotk2(numk2,denk2,ksum2)/(mp.sqrt(denk2)) + dkcoef2*k1dotk2(numk2,denk2,ksum2)**3/(denk2*mp.sqrt(denk2)))
				term += -m1*(numk2*delta_coef*k1dotk2(numk2,denk2,ksum2)/(mp.sqrt(denk2)) + dkcoef*k1dotk2(numk2,denk2,ksum2)**3/(denk2*mp.sqrt(denk2)))
			else:
				print('exceeded calculable power')

		elif kqexp == 4:
			# print('using power 4')	
			if q2exp == 0:
				coef1, coef2, coef3 = num_four_pow(d1,d2,denk2,m1,m2)
				term = coef1*numk2**2 + numk2*k1dotk2(numk2,denk2,ksum2)**2*coef2/denk2 + coef3*k1dotk2(numk2,denk2,ksum2)**4/denk2**2
			else:
				print(kqexp, q2exp, 'kqexp, q2exp')
				print('exceeded calculable power')
				# print('term at the end', term)

		if kqexp > 4:
			print(kqexp, q2exp, 'kqexp, q2exp')
			print('exceeded calculable power')
		
		res_list += term*term_list[i]*numk2**(k2exp)
		mp.nprint((term*term_list[i]*numk2**(k2exp), 'k2exp, q2exp, kqexp: ', exp_list[i], 'coef: ', term_list[i]),5)
	return res_list

def num_one_pow(d1,d2,denk2,m1,m2):
	#integral of k.q/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by sqrt(k^2)
	# coef in front of k_i
	coef = BubN(d1,d2-1,denk2,m1,m2) - BubN(d1-1,d2,denk2,m1,m2) - (denk2 + m2 - m1)*BubN(d1,d2,denk2,m1,m2)
	coef = coef/(2*denk2)
	return coef

def num_two_pow(d1,d2,denk2,m1,m2):
	#integral of (k.q)^2/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^2
	#denk2 are magnitudes of external momenta
	coef1 = -(BubN(d1,d2-2,denk2,m1,m2) - 2*(denk2 + m2 - m1)*BubN(d1,d2-1,denk2,m1,m2) + (denk2 + m2 - m1)**2*BubN(d1,d2,denk2,m1,m2)
		-2*BubN(d1-1,d2-1,denk2,m1,m2) + 2*(denk2 + m2 - m1)*BubN(d1-1,d2,denk2,m1,m2) + BubN(d1-2,d2,denk2,m1,m2))/(8*denk2) + BubN(d1-1,d2,denk2,m1,m2)/2 - m1*BubN(d1,d2,denk2,m1,m2)/2

	coef2 = BubN(d1-1,d2,denk2,m1,m2) - m1*BubN(d1,d2,denk2,m1,m2) - 3*coef1
	return coef1, coef2

def num_three_pow(d1,d2,denk2,m1,m2):
	#integral of (k.q)^3/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^3
	coef1 = 3/(16 * denk2*mp.sqrt(denk2))*(BubN(d1-3,d2,denk2,m1,m2)-3*BubN(d1-2,d2-1,denk2,m1,m2) - 4*denk2*BubN(d1-2,d2,denk2,m1,m2)
		+ 3*(denk2 + m2 - m1)*BubN(d1-2,d2,denk2,m1,m2) + 3*BubN(d1-1,d2-2,denk2,m1,m2)
		+ 4*denk2*BubN(d1-1,d2-1,denk2,m1,m2) - 6*(denk2 + m2 - m1)*BubN(d1-1,d2-1,denk2,m1,m2)
		-4*denk2*(denk2 + m2 - m1)*BubN(d1-1,d2,denk2,m1,m2) + 3*(denk2 + m2 - m1)**2*BubN(d1-1,d2,denk2,m1,m2)
		+ 4* denk2*m1*BubN(d1-1,d2,denk2,m1,m2) - BubN(d1,d2-3,denk2,m1,m2) + 3*(denk2 + m2 - m1)*BubN(d1,d2-2,denk2,m1,m2)
		-3*(denk2 + m2 - m1)**2*BubN(d1,d2-1,denk2,m1,m2) - 4*denk2*m1*BubN(d1,d2-1,denk2,m1,m2)
		+(denk2 + m2 - m1)**3*BubN(d1,d2,denk2,m1,m2) + 4*denk2*(denk2 + m2 - m1)*m1*BubN(d1,d2,denk2,m1,m2))
	
	coef2 = 1/(2*mp.sqrt(denk2))*(BubN(d1-1,d2-1,denk2,m1,m2) - BubN(d1-2,d2,denk2,m1,m2)
		-(denk2 + m2 - m1)*BubN(d1-1,d2,denk2,m1,m2) + m1*BubN(d1-1,d2,denk2,m1,m2) - m1*BubN(d1,d2-1,denk2,m1,m2)
		+(denk2 + m2 - m1)*m1*BubN(d1,d2,denk2,m1,m2))-5*coef1/3
	return coef1, coef2

def num_four_pow(d1,d2,denk2,m1,m2):
	coef1 = 3*(BubN(-4 + d1, d2, denk2, m1, m2) - 
		4*BubN(-3 + d1, -1 + d2, denk2, m1, m2) - 
		4*denk2*BubN(-3 + d1, d2, denk2, m1, m2) - 
		4*m1*BubN(-3 + d1, d2, denk2, m1, m2) + 
		4*m2*BubN(-3 + d1, d2, denk2, m1, m2) + 
		6*BubN(-2 + d1, -2 + d2, denk2, m1, m2) + 
		4*denk2*BubN(-2 + d1, -1 + d2, denk2, m1, m2) + 
		12*m1*BubN(-2 + d1, -1 + d2, denk2, m1, m2) - 
		12*m2*BubN(-2 + d1, -1 + d2, denk2, m1, m2) + 
		6*denk2**2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		12*denk2*m1*BubN(-2 + d1, d2, denk2, m1, m2) + 
		6*m1**2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		4*denk2*m2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		12*m1*m2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		6*m2**2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		4*BubN(-1 + d1, -3 + d2, denk2, m1, m2) + 
		4*denk2*BubN(-1 + d1, -2 + d2, denk2, m1, m2) - 
		12*m1*BubN(-1 + d1, -2 + d2, denk2, m1, m2) + 
		12*m2*BubN(-1 + d1, -2 + d2, denk2, m1, m2) + 
		4*denk2**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		8*denk2*m1*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		12*m1**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		8*denk2*m2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) + 
		24*m1*m2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		12*m2**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		4*denk2**3*BubN(-1 + d1, d2, denk2, m1, m2) - 
		12*denk2**2*m1*BubN(-1 + d1, d2, denk2, m1, m2) - 
		12*denk2*m1**2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		4*m1**3*BubN(-1 + d1, d2, denk2, m1, m2) - 
		4*denk2**2*m2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		8*denk2*m1*m2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		12*m1**2*m2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		4*denk2*m2**2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		12*m1*m2**2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		4*m2**3*BubN(-1 + d1, d2, denk2, m1, m2) + 
		BubN(d1, -4 + d2, denk2, m1, m2) - 
		4*denk2*BubN(d1, -3 + d2, denk2, m1, m2) + 
		4*m1*BubN(d1, -3 + d2, denk2, m1, m2) - 
		4*m2*BubN(d1, -3 + d2, denk2, m1, m2) + 
		6*denk2**2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		4*denk2*m1*BubN(d1, -2 + d2, denk2, m1, m2) + 
		6*m1**2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		12*denk2*m2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		12*m1*m2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		6*m2**2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		4*denk2**3*BubN(d1, -1 + d2, denk2, m1, m2) - 
		4*denk2**2*m1*BubN(d1, -1 + d2, denk2, m1, m2) + 
		4*denk2*m1**2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		4*m1**3*BubN(d1, -1 + d2, denk2, m1, m2) - 
		12*denk2**2*m2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		8*denk2*m1*m2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		12*m1**2*m2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		12*denk2*m2**2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		12*m1*m2**2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		4*m2**3*BubN(d1, -1 + d2, denk2, m1, m2) + 
		denk2**4*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2**3*m1*BubN(d1, d2, denk2, m1, m2) + 
		6*denk2**2*m1**2*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2*m1**3*BubN(d1, d2, denk2, m1, m2) + 
		m1**4*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2**3*m2*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2**2*m1*m2*BubN(d1, d2, denk2, m1, m2) - 
		4*denk2*m1**2*m2*BubN(d1, d2, denk2, m1, m2) - 
		4*m1**3*m2*BubN(d1, d2, denk2, m1, m2) + 
		6*denk2**2*m2**2*BubN(d1, d2, denk2, m1, m2) - 
		4*denk2*m1*m2**2*BubN(d1, d2, denk2, m1, m2) + 
		6*m1**2*m2**2*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2*m2**3*BubN(d1, d2, denk2, m1, m2) - 
		4*m1*m2**3*BubN(d1, d2, denk2, m1, m2) + 
		m2**4*BubN(d1, d2, denk2, m1, m2))/(128*denk2**2)
	coef2 = -3*(5*BubN(-4 + d1, d2, denk2, m1, m2) - 
		20*BubN(-3 + d1, -1 + d2, denk2, m1, m2) - 
		4*denk2*BubN(-3 + d1, d2, denk2, m1, m2) - 
		20*m1*BubN(-3 + d1, d2, denk2, m1, m2) + 
		20*m2*BubN(-3 + d1, d2, denk2, m1, m2) + 
		30*BubN(-2 + d1, -2 + d2, denk2, m1, m2) - 
		12*denk2*BubN(-2 + d1, -1 + d2, denk2, m1, m2) + 
		60*m1*BubN(-2 + d1, -1 + d2, denk2, m1, m2) - 
		60*m2*BubN(-2 + d1, -1 + d2, denk2, m1, m2) - 
		2*denk2**2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		12*denk2*m1*BubN(-2 + d1, d2, denk2, m1, m2) + 
		30*m1**2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		12*denk2*m2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		60*m1*m2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		30*m2**2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		20*BubN(-1 + d1, -3 + d2, denk2, m1, m2) + 
		36*denk2*BubN(-1 + d1, -2 + d2, denk2, m1, m2) - 
		60*m1*BubN(-1 + d1, -2 + d2, denk2, m1, m2) + 
		60*m2*BubN(-1 + d1, -2 + d2, denk2, m1, m2) - 
		12*denk2**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) + 
		24*denk2*m1*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		60*m1**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		72*denk2*m2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) + 
		120*m1*m2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		60*m2**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		4*denk2**3*BubN(-1 + d1, d2, denk2, m1, m2) + 
		4*denk2**2*m1*BubN(-1 + d1, d2, denk2, m1, m2) - 
		12*denk2*m1**2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		20*m1**3*BubN(-1 + d1, d2, denk2, m1, m2) + 
		12*denk2**2*m2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		24*denk2*m1*m2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		60*m1**2*m2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		36*denk2*m2**2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		60*m1*m2**2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		20*m2**3*BubN(-1 + d1, d2, denk2, m1, m2) + 
		5*BubN(d1, -4 + d2, denk2, m1, m2) - 
		20*denk2*BubN(d1, -3 + d2, denk2, m1, m2) + 
		20*m1*BubN(d1, -3 + d2, denk2, m1, m2) - 
		20*m2*BubN(d1, -3 + d2, denk2, m1, m2) + 
		30*denk2**2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		36*denk2*m1*BubN(d1, -2 + d2, denk2, m1, m2) + 
		30*m1**2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		60*denk2*m2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		60*m1*m2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		30*m2**2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		20*denk2**3*BubN(d1, -1 + d2, denk2, m1, m2) + 
		12*denk2**2*m1*BubN(d1, -1 + d2, denk2, m1, m2) - 
		12*denk2*m1**2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		20*m1**3*BubN(d1, -1 + d2, denk2, m1, m2) - 
		60*denk2**2*m2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		72*denk2*m1*m2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		60*m1**2*m2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		60*denk2*m2**2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		60*m1*m2**2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		20*m2**3*BubN(d1, -1 + d2, denk2, m1, m2) + 
		5*denk2**4*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2**3*m1*BubN(d1, d2, denk2, m1, m2) - 
		2*denk2**2*m1**2*BubN(d1, d2, denk2, m1, m2) + 
		4*denk2*m1**3*BubN(d1, d2, denk2, m1, m2) + 
		5*m1**4*BubN(d1, d2, denk2, m1, m2) + 
		20*denk2**3*m2*BubN(d1, d2, denk2, m1, m2) - 
		12*denk2**2*m1*m2*BubN(d1, d2, denk2, m1, m2) + 
		12*denk2*m1**2*m2*BubN(d1, d2, denk2, m1, m2) - 
		20*m1**3*m2*BubN(d1, d2, denk2, m1, m2) + 
		30*denk2**2*m2**2*BubN(d1, d2, denk2, m1, m2) - 
		36*denk2*m1*m2**2*BubN(d1, d2, denk2, m1, m2) + 
		30*m1**2*m2**2*BubN(d1, d2, denk2, m1, m2) + 
		20*denk2*m2**3*BubN(d1, d2, denk2, m1, m2) - 
		20*m1*m2**3*BubN(d1, d2, denk2, m1, m2) + 
		5*m2**4*BubN(d1, d2, denk2, m1, m2))/(64*denk2**2)
	coef3 = -(-35*BubN(-4 + d1, d2, denk2, m1, m2) + 
		140*BubN(-3 + d1, -1 + d2, denk2, m1, m2) - 
		20*denk2*BubN(-3 + d1, d2, denk2, m1, m2) + 
		140*m1*BubN(-3 + d1, d2, denk2, m1, m2) - 
		140*m2*BubN(-3 + d1, d2, denk2, m1, m2) - 
		210*BubN(-2 + d1, -2 + d2, denk2, m1, m2) + 
		180*denk2*BubN(-2 + d1, -1 + d2, denk2, m1, m2) - 
		420*m1*BubN(-2 + d1, -1 + d2, denk2, m1, m2) + 
		420*m2*BubN(-2 + d1, -1 + d2, denk2, m1, m2) - 
		18*denk2**2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		60*denk2*m1*BubN(-2 + d1, d2, denk2, m1, m2) - 
		210*m1**2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		180*denk2*m2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		420*m1*m2*BubN(-2 + d1, d2, denk2, m1, m2) - 
		210*m2**2*BubN(-2 + d1, d2, denk2, m1, m2) + 
		140*BubN(-1 + d1, -3 + d2, denk2, m1, m2) - 
		300*denk2*BubN(-1 + d1, -2 + d2, denk2, m1, m2) + 
		420*m1*BubN(-1 + d1, -2 + d2, denk2, m1, m2) - 
		420*m2*BubN(-1 + d1, -2 + d2, denk2, m1, m2) + 
		180*denk2**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		360*denk2*m1*BubN(-1 + d1, -1 + d2, denk2, m1, m2) + 
		420*m1**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) + 
		600*denk2*m2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		840*m1*m2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) + 
		420*m2**2*BubN(-1 + d1, -1 + d2, denk2, m1, m2) - 
		20*denk2**3*BubN(-1 + d1, d2, denk2, m1, m2) + 
		36*denk2**2*m1*BubN(-1 + d1, d2, denk2, m1, m2) - 
		60*denk2*m1**2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		140*m1**3*BubN(-1 + d1, d2, denk2, m1, m2) - 
		180*denk2**2*m2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		360*denk2*m1*m2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		420*m1**2*m2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		300*denk2*m2**2*BubN(-1 + d1, d2, denk2, m1, m2) + 
		420*m1*m2**2*BubN(-1 + d1, d2, denk2, m1, m2) - 
		140*m2**3*BubN(-1 + d1, d2, denk2, m1, m2) - 
		35*BubN(d1, -4 + d2, denk2, m1, m2) + 
		140*denk2*BubN(d1, -3 + d2, denk2, m1, m2) - 
		140*m1*BubN(d1, -3 + d2, denk2, m1, m2) + 
		140*m2*BubN(d1, -3 + d2, denk2, m1, m2) - 
		210*denk2**2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		300*denk2*m1*BubN(d1, -2 + d2, denk2, m1, m2) - 
		210*m1**2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		420*denk2*m2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		420*m1*m2*BubN(d1, -2 + d2, denk2, m1, m2) - 
		210*m2**2*BubN(d1, -2 + d2, denk2, m1, m2) + 
		140*denk2**3*BubN(d1, -1 + d2, denk2, m1, m2) - 
		180*denk2**2*m1*BubN(d1, -1 + d2, denk2, m1, m2) + 
		180*denk2*m1**2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		140*m1**3*BubN(d1, -1 + d2, denk2, m1, m2) + 
		420*denk2**2*m2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		600*denk2*m1*m2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		420*m1**2*m2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		420*denk2*m2**2*BubN(d1, -1 + d2, denk2, m1, m2) - 
		420*m1*m2**2*BubN(d1, -1 + d2, denk2, m1, m2) + 
		140*m2**3*BubN(d1, -1 + d2, denk2, m1, m2) - 
		35*denk2**4*BubN(d1, d2, denk2, m1, m2) + 
		20*denk2**3*m1*BubN(d1, d2, denk2, m1, m2) - 
		18*denk2**2*m1**2*BubN(d1, d2, denk2, m1, m2) + 
		20*denk2*m1**3*BubN(d1, d2, denk2, m1, m2) - 
		35*m1**4*BubN(d1, d2, denk2, m1, m2) - 
		140*denk2**3*m2*BubN(d1, d2, denk2, m1, m2) + 
		180*denk2**2*m1*m2*BubN(d1, d2, denk2, m1, m2) - 
		180*denk2*m1**2*m2*BubN(d1, d2, denk2, m1, m2) + 
		140*m1**3*m2*BubN(d1, d2, denk2, m1, m2) - 
		210*denk2**2*m2**2*BubN(d1, d2, denk2, m1, m2) + 
		300*denk2*m1*m2**2*BubN(d1, d2, denk2, m1, m2) - 
		210*m1**2*m2**2*BubN(d1, d2, denk2, m1, m2) - 
		140*denk2*m2**3*BubN(d1, d2, denk2, m1, m2) + 
		140*m1*m2**3*BubN(d1, d2, denk2, m1, m2) - 
		35*m2**4*BubN(d1, d2, denk2, m1, m2))/(128*denk2**2)
	return coef1, coef2, coef3




@lru_cache(None)
def tri_dim_two(n1,n2,d1,numk21,numk22,ksum2,dm):
	# integral of (k1 - q)^2^n1 (k2 + q)^2^n2/(q2+dm)^d1

	# term_list1 are the coefficients of (k1 - q)^2^n1 corresponding to the exponents in exp_list   
	# exp_list1 are the exponents of (k1 - q)^2^n1 of the form k1^2^k2exp1*q^2^q2exp1*(k.q)^kqexp1, 
	# written as (k2exp1, q2exp1, kqexp1) 

	# term_list2 are the coefficients of (k2 + q)^2^n2 corresponding to the exponents in exp_list   
	# exp_list2 are the exponents of (k2 + q)^2^n2 of the form k2^2^k2exp2*q^2^q2exp2*(k.q)^kqexp2, 
	# written as (k2exp2, q2exp2, kqexp2) 

	term_list1, exp_list1 = num_terms(n1,True)
	term_list2, exp_list2 = num_terms(n2,False)
	res_list = 0

	for i1 in range(len(exp_list1)):
		for i2 in range(len(exp_list2)):
			k2exp1 = exp_list1[i1][0]
			k2exp2 = exp_list2[i2][0]
			q2exp = exp_list1[i1][1] + exp_list2[i2][1]
			kqexp1 = exp_list1[i1][2]
			kqexp2 = exp_list2[i2][2]
			kqexp = kqexp1 + kqexp2

			term = 0
			if kqexp%2 == 0:
				# if kqexp is odd then the integral vanishes by symmetry q -> -q				
				if kqexp == 6:
					print('using power 6')				
				if kqexp != 0:
					#cases where kqexp == 2
					if kqexp1 == 2 and kqexp2 == 0:
						term = dim_gen(q2exp+1,d1,dm)*(numk21)**(kqexp1/2)/3
					elif kqexp1 == 0 and kqexp2 == 2:
						term = dim_gen(q2exp+1,d1,dm)*(numk22)**(kqexp2/2)/3
					elif kqexp1 == 1 and kqexp2 == 1:
						term = dim_gen(q2exp+1,d1,dm)*(k1dotk2(numk21,numk22,ksum2))/3
						# mp.nprint((term,'test'))

					# cases where kqexp == 4
					elif kqexp1 == 0 and kqexp2 == 4:
						term = term = dim_gen(q2exp+2,d1,dm)*(numk22**2)/5	
						# nprint((term, 'kqexp = 4, kqexp2 = 4, kqexp1 = 0'))
					elif kqexp1 == 4 and kqexp2 == 0:
						term = dim_gen(q2exp+2,d1,dm)*(numk21**2)/5
						# nprint((term, 'kqexp = 4, kqexp1 = 4, kqexp2 = 0', q2exp))
					elif kqexp1 == 1 and kqexp2 == 3:
						term = dim_gen(q2exp+2,d1,dm)*(k1dotk2(numk21,numk22,ksum2)*numk22)/5
					elif kqexp1 == 3 and kqexp2 == 1:
						term = dim_gen(q2exp+2,d1,dm)*(k1dotk2(numk21,numk22,ksum2)*numk21)/5
					elif kqexp1 == 2 and kqexp2 == 2:
						term = dim_gen(q2exp+2,d1,dm)*(numk21*numk22 + 2*(k1dotk2(numk21,numk22,ksum2))**2)/15


					# cases where kqexp == 6
					elif kqexp == 6 and kqexp1 == 4:
						term = dim_gen(q2exp+3,d1,dm)*(numk21**2*numk22 + 4*(k1dotk2(numk21,numk22,ksum2))**2*numk21)/35
						nprint((term, 'kqexp = 6, kqexp1 = 4, kqexp2 = 2'))
					elif kqexp == 6 and kqexp1 == 3:
						term = dim_gen(q2exp+3,d1,dm)*(3*numk21*numk22*k1dotk2(numk21,numk22,ksum2) + 2*(k1dotk2(numk21,numk22,ksum2))**3)/35
						nprint((term, 'kqexp = 6, kqexp1 = 3, kqexp2 = 3', q2exp))
					elif kqexp == 6 and kqexp2 == 4:
						term = dim_gen(q2exp+3,d1,dm)*(numk22**2*numk21 + 4*(k1dotk2(numk21,numk22,ksum2))**2*numk22)/35
						nprint((term, 'kqexp = 6, kqexp2 = 4, kqexp1 = 2'))
					else:
						print('ERROR: case not considered', kqexp, q2exp, kqexp1, kqexp2)		
				else:
					# case where kqexp == 0
					term = dim_gen(q2exp,d1,dm)
			else:
				term = mpf('0')

			res_list += term*term_list2[i2]*term_list1[i1]*(numk21)**(k2exp1)*(numk22)**(k2exp2)
	return res_list

def k1dotk2(k21,k22,ksum2):
	return (ksum2 - k21 - k22)/2

@lru_cache(None)
def Ltrian(n1, d1, n2, d2, n3, d3, 
		   k21, k22, k23, m1, m2, m3):
	
	if n1 == 0 and n2 == 0 and n3 == 0:
		# mp.nprint((TriaN(d1,d2,d3,k21,k22,k23,m1,m2,m3)),6)
		return TriaN(d1,d2,d3,k21,k22,k23,m1,m2,m3)
	if d1 == 0 and n1 != 0:
		return Ltrian(0,-n1,n2,d2,n3,d3,k21,k22,k23,mpf('0'),m2,m3)
	if d2 == 0 and n2 != 0:
		return Ltrian(n1,d1,0,-n2,n3,d3,k21,k22,k23,m1,mpf('0'),m3)
	if d3 == 0 and n3 != 0:
		return Ltrian(n1,d1,n2,d2,0,-n3,k21,k22,k23,m1,m2,mpf('0'))
	if n1 > 0:
		return Ltrian(n1-1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3) - m1*Ltrian(n1-1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3)
	if n2 > 0:
		return Ltrian(n1,d1,n2-1,d2-1,n3,d3,k21,k22,k23,m1,m2,m3) - m2*Ltrian(n1,d1,n2-1,d2,n3,d3,k21,k22,k23,m1,m2,m3)
	if n3 > 0:
		return Ltrian(n1,d1,n2,d2,n3-1,d3-1,k21,k22,k23,m1,m2,m3) - m3*Ltrian(n1,d1,n2,d2,n3-1,d3,k21,k22,k23,m1,m2,m3)
	if n1 < 0:
		# print('neg n1')
		# print(n1,d1-1,n2,d2,n3,d3, Ltrian(n1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3), 'Ltrian 1')
		# print(n1+1,d1,n2,d2,n3,d3, Ltrian(n1+1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3), 'Ltrian 2')
		return (1/m1)*(Ltrian(n1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3) - Ltrian(n1+1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3))
	if n2 < 0:
		# print('neg n2')
		# print(n1,d1,n2,d2-1,n3,d3, Ltrian(n1,d1,n2,d2-1,n3,d3,k21,k22,k23,m1,m2,m3), 'Ltrian 1')
		# print(n1,d1,n2+1,d2,n3,d3, Ltrian(n1,d1,n2+1,d2,n3,d3,k21,k22,k23,m1,m2,m3), 'Ltrian 2')
		return (1/m2)*(Ltrian(n1,d1,n2,d2-1,n3,d3,k21,k22,k23,m1,m2,m3) - Ltrian(n1,d1,n2+1,d2,n3,d3,k21,k22,k23,m1,m2,m3))
	if n3 < 0:
		# print('neg n3')
		# print(n1,d1,n2,d2,n3,d3-1, Ltrian(n1,d1,n2,d2,n3,d3-1,k21,k22,k23,m1,m2,m3), 'Ltrian 1')
		# print(n1,d1,n2,d2,n3+1,d3, Ltrian(n1,d1,n2,d2,n3+1,d3,k21,k22,k23,m1,m2,m3), 'Ltrian 2')
		return (1/m3)*(Ltrian(n1,d1,n2,d2,n3,d3-1,k21,k22,k23,m1,m2,m3) - Ltrian(n1,d1,n2,d2,n3+1,d3,k21,k22,k23,m1,m2,m3))


#@lru_cache(None)
def BubMaster(k2, M1, M2):	
	m1 = M1/k2
	m2 = M2/k2

	if im(1j*(1 + m1-m2-2)+2*sqrt(m1)) > 0 and im(1j*(1 + m1-m2)+2*sqrt(m2)) < 0:
		sign = 1
	else:
		sign = 0
	bubmaster = sqrt(pi())/(k2+0j)**(1/2)*(1j*log(1j*(1 + m1-m2-2)+2*sqrt(m1))-1j*log(1j*(1 + m1-m2)+2*sqrt(m2))+2*pi()*sign)
	return bubmaster



#some short useful functions for the Triangle Master integral

def Diakr(a, b, c):
	return b**2-4*a*c

def Prefactor(a,y1,y2):
	#computes prefactor that shows up in Fint

	if fabs(y1) > CHOP_TOL and fabs(y2) > CHOP_TOL: 
		return sqrt(-y1+0j)*sqrt(-y2+0j)/(sqrt(a*(y1+0j)*(y2+0j)))
	if fabs(y1) < CHOP_TOL and fabs(y2) > CHOP_TOL:
		return sqrt(-y2+0j)/sqrt(-a*(y2+0j))
	if fabs(y1) > CHOP_TOL and fabs(y2) < CHOP_TOL:
		return sqrt(-y1+0j)/sqrt(-a*(y1+0j))
	if fabs(y1) < CHOP_TOL and fabs(y2) < CHOP_TOL:
		return 1/sqrt(a)

def Antideriv(x, y1, y2, x0):
	if almosteq(x0,y2,CHOP_TOL):
		# case where x0 = y2 = 0 or 1
		if almosteq(x,y2,CHOP_TOL):
			return 0
		return 2*sqrt(x-y1)/(-x0+y1)/sqrt(x-y2)
	
	if fabs(x0-y1) < 10**(-GLOBAL_PREC):
		print('WARNING: switching var in Antideriv')
		#x0 = y2 = 0 or 1
		return Antideriv(x,y2,y1,x0)

	
	prefac = 2/(sqrt(-x0+y1+0j)*sqrt(x0-y2+0j))
	temp = sqrt(x-y1+0j)*sqrt(x0-y2+0j)/sqrt(-x0+y1)	
	# mp.nprint(("temp", temp,y1, y2, x, x0),5)
	if x == 1 and almosteq(1,y2,CHOP_TOL):
		LimArcTan = chop(1j * sqrt(-temp**2) * pi()/(2*temp), tol = CHOP_TOL)
		return  prefac * LimArcTan
	if x == 0 and almosteq(0,y2,CHOP_TOL):
		LimArcTan = chop(sqrt(temp**2) * pi()/(2*temp), tol = CHOP_TOL)
		return  prefac * LimArcTan
	mp.nprint(sqrt(x-y2),5)
	return prefac*atan(temp/sqrt(x-y2))
	

#Calculation of the Triangle Master integral 

def Fint(aa,y1,y2,x0):

	y1 = chop(y1, tol = CHOP_TOL)
	y2 = chop(y2, tol = CHOP_TOL)
	x0 = chop(x0, tol = CHOP_TOL)

	rey1 = re(y1)
	imy1 = chop(im(y1), tol = CHOP_TOL)
	rey2 = re(y2)
	imy2 = chop(im(y2), tol = CHOP_TOL)
	rex0 = re(x0)
	imx0 = chop(im(x0), tol = CHOP_TOL)

	c = imy1**2*imy2*rex0 - imy1*imy2**2*rex0-imx0**2*imy2*rey1 + imx0*imy2**2*rey1-imy2*rex0**2*rey1 + imy2*rex0*rey1**2 + imx0**2*imy1*rey2-imx0*imy1**2*rey2+imy1*rex0**2*rey2-imx0*rey1**2*rey2-imy1*rex0*rey2**2+imx0*rey1*rey2**2
	a = imy1*rex0-imy2*rex0-imx0*rey1+imy2*rey1+imx0*rey2-imy1*rey2
	b = -imx0**2*imy1 + imx0*imy1**2+imx0**2*imy2-imy1**2*imy2-imx0*imy2**2+imy1*imy2**2-imy1*rex0**2+imy2*rex0**2+imx0*rey1**2-imy2*rey1**2-imx0*rey2**2+imy1*rey2**2

# if x0 is real there will always be a crossing through i or -i, which gives a cut of pi/2 instead of pi
	derivcritx0 = 0
	signx0 = 0
	
	cutx0 = 0
	if 0 < rex0 < 1 and almosteq(imx0,0, CHOP_TOL):
		derivcritx0 = (y1 - y2)/2/sqrt(-(x0-y1)**2)/(x0-y2)		
		if re(derivcritx0) < 0:
			signx0 = 1
		else:
			signx0 = -1	
		cutx0 = signx0*pi()/(sqrt(-x0+y1+0j)*sqrt(x0-y2+0j))
	else:
		cutx0 = 0
		
# find remaining crossings of the imaginary axis
	numbranchpoints = 0
	sign = 0
	if almosteq(a,0):
		if b != 0:
			xsol = [- c / b]
		else:
			xsol = []
	else:
		if b**2-4*a*c > 0:
			xsol = [(-b + sqrt(b**2-4*a*c))/(2*a),(-b - sqrt(b**2-4*a*c))/(2*a)]
		else:
			#case where there is no intersection of the real axis (includes double zero)
			 xsol = []

	xsol = [x for x in xsol if x > CHOP_TOL and x < mpf(1)-CHOP_TOL and not(almosteq(x,x0,CHOP_TOL))]
	
	if len(xsol) > 0:
		#print(xsol,x0)
		xbranch = []
		atanarglist = [chop(sqrt(x-y1+0j)*sqrt(x0-y2 + 0j)/(sqrt(-x0+y1+0j)*sqrt(x-y2+0j)), tol = CHOP_TOL) for x in xsol]
		abscrit = [fabs(atanarg) for atanarg in atanarglist]
		argcrit = [arg(atanarg) for atanarg in atanarglist]
		for i in range(len(xsol)):
			if abscrit[i] > 1 and almosteq(fabs(argcrit[i]), pi()/2., CHOP_TOL):
				numbranchpoints += 1
				xbranch.append(xsol[i])

	if numbranchpoints == 1:
		derivcrit = [sqrt(x0-y2+0j)/sqrt(-x0+y1+0j)*(1/(2*sqrt(x-y1+0j)*sqrt(x-y2+0j)) -sqrt(x-y1+0j)/(2*(x-y2+0j)**(3/2))) for x in xbranch]
		if re(derivcrit[0]) < 0:
			sign = 1
		else:
			sign = -1
	else:
		sign = 0

	if sign == 0:
		cut = 0
	else:
		cut = sign*pi()*2/(sqrt(-x0+y1+0j)*sqrt(x0-y2+0j))
	
	prefac0 = chop(Prefactor(aa,y1,y2))
	# print("aa,y1,y2,x0",aa,y1,y2,x0)
	# print(numbranchpoints, cut, cutx0)
	# print(Antideriv(1.,y1,y2,x0) - Antideriv(0.,y1,y2,x0))
	# mp.nprint((Antideriv(mpf(1),y1,y2,x0) - Antideriv(mpf(0),y1,y2,x0)),5)
	result = prefac0*(sqrt(pi())/2)*(cut + cutx0 + Antideriv(mpf(1),y1,y2,x0) - Antideriv(mpf(0),y1,y2,x0))
	# mp.nprint(("result", Antideriv(mpf(1),y1,y2,x0) - Antideriv(mpf(0),y1,y2,x0),Antideriv(mpf(1),y1,y2,x0) , Antideriv(mpf(0),y1,y2,x0)),5)
	return result

#@lru_cache(None)
def TrMxy(y, k21, k22, k23, M1, M2, M3):

	# print('y',y)
	Num1 = 4*k22*y+2*k21-2*k22-2*k23
	Num0 = -4*k22*y+2*M2-2*M3+2*k22
	DeltaR2 = -k21*y+k23*y-k23
	DeltaR1 = -M2*y+M3*y+k21*y-k23*y+M1-M3+k23
	DeltaR0 = M2*y-M3*y+M3
	DeltaS2 = -k21**2+2*k21*k22+2*k21*k23-k22**2+2*k22*k23-k23**2
	DeltaS1 =-4*M1*k22-2*M2*k21+2*M2*k22+2*M2*k23+2*M3*k21+2*M3*k22-2*M3*k23-2*k21*k22+2*k22**2-2*k22*k23
	DeltaS0 =-M2**2+2*M2*M3-2*M2*k22-M3**2-2*M3*k22-k22**2

	DiakrS = chop(sqrt(Diakr(DeltaS2, DeltaS1, DeltaS0)), tol = CHOP_TOL)
	#print('Num1',Num1, 'Num0', Num0, 'DiakrS',DiakrS)
	solS1 = (-DeltaS1+DiakrS)/2/DeltaS2
	solS2 = (-DeltaS1-DiakrS)/2/DeltaS2  
	#print("solS1",solS1,"solS2", solS2)
	cf2 = chop(-(Num1*solS2+Num0)/DiakrS, tol = CHOP_TOL)
	cf1 = chop((Num1*solS1+Num0)/DiakrS , tol = CHOP_TOL)
		
	DiakrR = chop(sqrt(Diakr(DeltaR2, DeltaR1, DeltaR0)), tol = CHOP_TOL)
				  
	solR1 = ((-DeltaR1+DiakrR)/2)/DeltaR2     
	solR2 = ((-DeltaR1-DiakrR)/2)/DeltaR2 
	# print("cf1, cf2", cf1, cf2)
	# print('Fint cf2 = ',Fint(DeltaR2, solR1, solR2, solS2))
	# print('Fint cf1 = ',Fint(DeltaR2, solR1, solR2, solS1))
	# print("cf1, cf2", cf1, cf2)
	# print('Fint cf2 = ',Fint(DeltaR2, solR1, solR2, solS2))
	# print('Fint cf1 = ',Fint(DeltaR2, solR1, solR2, solS1))
	if fabs(cf1) < CHOP_TOL:
		# print('neglect cf1')
		# print('Fint = ',cf2*Fint(DeltaR2, solR1, solR2, solS2))
		return cf2*Fint(DeltaR2, solR1, solR2, solS2)
	elif fabs(cf2) < CHOP_TOL:
		# print('neglect cf2')
		# print('Fint = ',cf1*Fint(DeltaR2, solR1, solR2, solS1))
		return cf1*Fint(DeltaR2, solR1, solR2, solS1)
	else:
		# print('cf_i Fint = ',cf2*Fint(DeltaR2, solR1, solR2, solS2)+cf1*Fint(DeltaR2, solR1, solR2, solS1))
		# mp.nprint((Fint(DeltaR2, solR1, solR2, solS2)),5)
		return cf2*Fint(DeltaR2, solR1, solR2, solS2)+cf1*Fint(DeltaR2, solR1, solR2, solS1)

#@lru_cache(None)
def TriaMasterZeroMasses(k21, k22, k23):
	#case for triangle integrals where all masses vanish
	return pi()**(3/2)/sqrt(k21)/sqrt(k22)/sqrt(k23)

#@lru_cache(None)
def TriaMaster(k21, k22, k23, M1, M2, M3):
	#--- masses are squared
	if M1 == 0 and M2 == 0 and M3 == 0:
		return  TriaMasterZeroMasses(k21, k22, k23)
	else:
		# mp.nprint((TrMxy(mpf(1), k21, k22,k23, M1, M2, M3)), 5)
		triamaster = chop(TrMxy(mpf(1), k21, k22,k23, M1, M2, M3)-TrMxy(mpf(0), k21, k22,k23,M1, M2, M3), tol = CHOP_TOL)
		return triamaster

