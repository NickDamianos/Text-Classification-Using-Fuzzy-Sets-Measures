import numpy as np

'''
	Miscellaneous measures for (intuitionistic) fuzzy sets
	========================================================================
 	@author: nikolaos damianos
	© Visual Computing Group, HUMAIN-Lab, Eastern Macedonia and Thrace Institute of Technology
	01-Mar-2017 - Release
'''
def Miscellaneous():
    return ['tamalikaDivergence', 'tamalikaDivergence2', 'tamalikaDivergence3']

#####################################################################################################################

def miscs(name,A,B=None , k=0.5 , x=None , y=None):
    if name == 'tamalikaDivergence':
        return __fuzzyDivergence(A,B)
    elif name == 'tamalikaDivergence2':
        return __fuzzyIndex(A,k)
    elif name == 'tamalikaDivergence3':
        return __fuzzyGeometry(A,x,y)
    else:
        print('Unknown divergence measure input.')
        return None

########################################################################################################################

def __fuzzyDivergence(A , B):
    '''
	FUZZYDIVERGENCE: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Tamalika Chaira, A.K. Ray, from the related article 
	"Threshold selection using fuzzy set theory"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		D: Distance measure.
    '''

    A = np.array(A)
    A_m = np.array(A[:, 0])

    B = np.array(B)
    B_m = np.array(B[:, 0])

    if (len(A[0]) != 2 or len(B[0]) != 2) and (A.shape != B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    ptr1 = (np.subtract(1 , A_m ) + B_m) * np.exp(A_m - B_m)
    ptr2 = (np.subtract(1 , B_m) + A_m) * np.exp(B_m - A_m)

    D = np.subtract(2 , ptr1 - ptr2)

    return np.sum(D)

########################################################################################################################

def __fuzzyIndex(A,k):
    '''
	FUZZYGEOMETRY: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Tamalika Chaira, A.K. Ray, from the related article 
	"Threshold selection using fuzzy set theory"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		x, y: dimensions of original image.
	OUTPUT:
		D: Distance measure.
	'''
    A = np.array(A)
    A_m = np.array(A[:,0])


    if not (k == 1) and (not (k == 0.5)):
        raise ValueError('k must be equal to 0.5 or 1 only')

    ff = 2.0 / float(np.power(len(A), k))

    jj = np.subtract(1 , A_m)

    kk  = np.minimum(A_m , jj)

    yy = np.sum(np.power(kk,(1.0/float(k))))

    return ff * yy**k

########################################################################################################################

def __fuzzyGeometry(A,x,y):
    '''
	FUZZYINDEX: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Tamalika Chaira, A.K. Ray, from the related article 
	"Threshold selection using fuzzy set theory"
	
	INPUTS:
		A: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		k: 0.5 for quadratic index or 1 for linear index 
	OUTPUT:
		D: Distance measure.
    '''

    A = np.array(A)
    mx = np.array(A[:, 0])
	
    if x == None :
        x = mx.shape[0]

    if y == None:
        y = mx.shape[1]
	
    if x > y:
        x = y
    else:
        y = x

    mx.resize((x,y))

    ptr1 = np.sum(np.absolute(mx[:, :len(mx[0]) - 1] - mx[:, 1:]))
    ptr2 = np.sum(np.absolute(mx[:len(mx[0]) - 1, :] - mx[1:, :]))
    Pm = ptr1 + ptr2

    am = np.sum(mx)

    if Pm != 0:
        pf = am / float(Pm ** 2)
    else:
        pf = 0
    return pf

########################################################################################################################
