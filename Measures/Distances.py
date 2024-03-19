import numpy as np

'''
	Distance measures for (intuitionistic) fuzzy sets
	========================================================================
	© Visual Computing Group, HUMAIN-Lab, Eastern Macedonia and Thrace Institute of Technology
	01-Mar-2017 - Release
'''
def Distances_():
    return ['defFuzzyDistance',
           'defIFSDistance',
           'wangXinDistance',
           'yangChiclanaDistance',
           'grzegorDistance',
           'vlachSergDistance']

#####################################################################################################################

def distance(distanceMeasure, A, B, type = None, p = 1, w = None ): #,k = None, x = None, y = None):
    if distanceMeasure == 'defFuzzyDistance':
        return __IFS_Dist00(A, B ,type)
    elif distanceMeasure == 'defIFSDistance':
        return __IFS_Dist01(A,B,type)
    elif distanceMeasure == 'wangXinDistance':
        return __IFS_Dist02(A, B, type, w, p)
    elif distanceMeasure == 'yangChiclanaDistance':
        return __IFS_Dist03(A,B, type)
    elif distanceMeasure == 'grzegorDistance':
        return __IFS_Dist04(A, B, type)
    elif distanceMeasure == 'vlachSergDistance':
        return __IFS_Dist05(A, B)
    else:
        print (distanceMeasure)
        raise ValueError('Unknown distance measure input.')
        

########################################################################################################################

def __IFS_Dist00(A, B, type):
    '''
	IFS_DIST00: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by K.T. Atanassov, from the related article 
	"Distances between intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed distance:
			'H' for Hamming,
			'E' for Euclidean,
			'nH' for normalized Hamming and
			'nE' for normalized Euclidean.
	OUTPUT:
		D: Distance measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    '''
    A_m.flags.writeable = False
    A_v.flags.writeable = False
    B_m.flags.writeable = False
    B_v.flags.writeable = False
    '''
    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError ('A and B parameters must be 2-D Nx2 matrices.')
  

    if type == 'H' or type == 'h':
        ptr1 = np.absolute(A_m - B_m)
        ptr2 = np.absolute(A_v - B_v)
        
        D = np.sum(ptr1 + ptr2)
        
        return D / 2.0
    elif type == 'E' or type == 'e':
        ptr1 = np.power(A_m - B_m, 2.0)
        ptr2 = np.power(A_v - B_v, 2.0)
        ptrs = np.sum(ptr1 + ptr2)
        D = ptrs / 2.0

        return np.sqrt(D)
    elif type == 'nH' or type == 'nh':
        ptr1 = np.absolute(A_m - B_m)
        ptr2 = np.absolute(A_v - B_v)
        D = np.sum(ptr1 + ptr2)

        return D / (2.0 * len(A))

    elif type == 'nE' or type == 'ne':
        ptr1 = np.power(A_m - B_m, 2.0)
        ptr2 = np.power(A_v - B_v, 2.0)
        ptrs = np.sum(ptr1 + ptr2)
        D = ptrs / (2.0 * len(A))

        return np.sqrt(D)
    else:
        raise ValueError('Distance type parameter must be H, E, nH or nE.')
        

########################################################################################################################

def __IFS_Dist01(A, B, type):
    '''
	IFS_DIST01: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by E. Szmidt and A_v. Kacprzyk, from the related article
	"Distances between intuitionistic fuzzy sets".
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed distance:
			'H' for Hamming,
			'E' for Euclidean,
			'nH' for normalized Hamming and
			'nE' for normalized Euclidean.
	OUTPUT:
		D: Distance measure.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])
    '''
    A_m.flags.writeable = False
    A_v.flags.writeable = False
    B_m.flags.writeable = False
    B_v.flags.writeable = False
    
    A_p.flags.writeable = False
    B_p.flags.writeable = False
    '''
    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError ('A and B parameters must be 2-D Nx3 matrices.')
        

    if type == 'H' or type == 'h':
        ptr1 = np.absolute(A_m - B_m)
        ptr2 = np.absolute(A_v - B_v)
        ptr3 = np.absolute(A_p - B_p)

        D = np.sum(ptr1 + ptr2 + ptr3)
        return D / 2.0
    elif type == 'E' or type == 'e':
        ptr1 = np.power(A_m - B_m, 2.0)
        ptr2 = np.power(A_v - B_v, 2.0)
        ptr3 = np.power(A_p - B_p, 2.0)

        D = np.sum(ptr1 + ptr2 + ptr3) / 2.0
        return np.sqrt(D)
    elif type == 'nH' or type == 'nh':
        ptr1 = np.absolute(A_m - B_m)
        ptr2 = np.absolute(A_v - B_v)
        ptr3 = np.absolute(A_p - B_p)

        D = np.sum(ptr1 + ptr2 + ptr3)
        return D / (2.0 * len(A))
    elif type == 'nE' or type == 'ne':
        ptr1 = np.power(A_m - B_m, 2.0)
        ptr2 = np.power(A_v - B_v, 2.0)
        ptr3 = np.power(A_p - B_p, 2.0)

        D = np.sum(ptr1 + ptr2 + ptr3) / (2.0 * len(A))
        return np.sqrt(D)
    else:
        raise ValueError('Distance type parameter must be H, E, nH or nE.')
        

########################################################################################################################

def __IFS_Dist02(A, B, type, w , p = 1):
    '''
	IFS_DIST02: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by W. Wang and X. Xin, from the related article 
	"Distance measure between intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed distance: 1 or 2.
		w: weights for the computed distance. If None then normalized weights are
		calculated
		p: must be a positive integer. Used only for type = 2.
	OUTPUT:
		D: Distance measure.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    '''
    A_m.flags.writeable = False
    A_v.flags.writeable = False
    B_m.flags.writeable = False
    B_v.flags.writeable = False
    '''
    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError ('A and B parameters must be 2-D Nx2 matrices.')
        
    if type == 2 and w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    elif w is not None:
        w = np.array(w)
        if type == 2 or (type == 1 and (w.size > 1 and w.sum() != 0)):
            if w.size != A.size:
                raise ValueError('Weight parameter dimensions must be 2-D Nx1 matrix.')
                
            if np.sum(w < 0) > 0 or np.sum(w > 1) > 1:
                raise ValueError('Weight values must be 0 <= w <= 1.')
                

    if type == 2 :
        if p < 1 or not isinstance(p, int):
            raise ValueError('p parameter must be integer')

    mDif = np.absolute(A_m - B_m)
    vDif = np.absolute(A_v - B_v)
    n = A.shape[0]

    if type == 1:
        part1 = (mDif + vDif) / 4.0
        part2 = np.maximum(mDif, vDif) / 2.0
        if w is None:
            return np.sum(part1 + part2) / float(n)
        else:
            return np.sum(w * part1 + part2) / float(np.sum(w))

    elif type == 2:
        fM = mDif  + vDif
        fM = fM / 2.0

        return (1.0 / (n ** (1.0 / p))) * np.sum((fM ** p)) ** (1.0 / p)#np.sum(w * fV ** p) ** (1.0 / p)

########################################################################################################################

def __IFS_Dist03(A, B ,type):
    '''
	IFS_DIST03: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Y. Yang and F. ChiclanA,B, from the related article 
	"Consistency of 2D and 3D distances of intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed distance:
			'H' for Hamming,
			'E' for Euclidean,
			'nH' for normalized Hamming and
			'nE' for normalized Euclidean.
	OUTPUT:
		D: Distance measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B_m = np.array(B[:,0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])
    '''
    A_m.flags.writeable = False
    A_v.flags.writeable = False
    B_m.flags.writeable = False
    B_v.flags.writeable = False
    
    A_p.flags.writeable = False
    B_p.flags.writeable = False
    '''
    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError ('A and B parameters must be 2-D Nx3 matrices.')
        

    mDif = A_m - B_m
    vDif = A_v - B_v
    pDif = A_p - B_p

    if type ==  'H':
        return np.sum(np.maximum(np.maximum(np.absolute(mDif), np.absolute(vDif)), np.absolute(pDif)))
    elif type ==  'E':
        return np.sqrt(np.sum(np.maximum(np.maximum(mDif ** 2, vDif ** 2), pDif ** 2)))
    elif type== 'nH':
        return np.sum(np.maximum(np.maximum(np.absolute(mDif), np.absolute(vDif)), np.absolute(pDif))) / float(len(A))
    elif type== 'nE':
        return np.sqrt(np.sum(np.maximum(np.maximum(mDif ** 2, vDif ** 2), pDif ** 2)) / float(len(A)))
    else:
        raise ValueError('Distance type parameter must be H, E, nH or nE.')
        

########################################################################################################################

def __IFS_Dist04(A, B ,type):
    '''
	IFS_DIST04: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by P. Grzegorzewski, from the related article 
	"Distances between intuitionistic fuzzy sets and/or interval-valued fuzzy 
	sets based on the Hausdorff metric"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed distance:
			'H' for Hamming,
			'E' for Euclidean,
			'nH' for normalized Hamming and
			'nE' for normalized Euclidean.
	OUTPUT:
		D: Distance measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    '''
    A_m.flags.writeable = False
    A_v.flags.writeable = False
    B_m.flags.writeable = False
    B_v.flags.writeable = False
    '''
    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError ('A and B parameters must be 2-D Nx2 matrices.')
        

    mDif = A_m - B_m
    vDif = A_v - B_v


    if type == 'H':
        return np.sum(np.maximum(np.absolute(mDif), np.absolute(vDif)))
    elif type == 'E':
        return np.sqrt(np.sum(np.maximum(mDif**2, vDif**2)))
    elif type == 'nH':
        return np.sum(np.maximum(np.absolute(mDif), np.absolute(vDif)))/float(len(A))
    elif type == 'nE':
        return np.sqrt(np.sum(np.maximum((mDif) ** 2, vDif **2))/float(len(A)))
    else:
        raise ValueError('Distance type parameter must be H, E, nH or nE.')
        

########################################################################################################################

def __IFS_Dist05(A, B):
    '''
	IFS_DIST05: Calculates distance between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by I.K. Vlachos, G.D. Sergiadis, from the related article 
	"Intuitionistic fuzzy information - Applications to pattern recognition"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
	 OUTPUT:
		D: Distance measure
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError ('A and B parameters must be 2-D Nx2 matrices.')
        
    '''
    A_m.flags.writeable = False
    A_v.flags.writeable = False
    B_m.flags.writeable = False
    B_v.flags.writeable = False
    '''
    
    n = A.shape[0]
	
    
	
    part1 = A_m/(1/2 * (A_m + B_m))
    part1 = np.where(part1>0, np.log(part1), part1)
    part2 = A_v/(1/2 * (A_v + B_v))
    part2 = np.where(part2>0, np.log(part2), part2)
    IAB = np.sum(A_m * part1 + A_v * part2)
	
    part1 = B_m/(1/2 * (B_m + A_m))
    part1 = np.where(part1>0, np.log(part1), part1)
    part2 = B_v/(1/2 * (B_v + A_v))
    part2 = np.where(part2>0, np.log(part2), part2)
    
	
    IBA = np.sum(B_m * part1 + B_v * part2)
    
    return IAB + IBA

########################################################################################################################