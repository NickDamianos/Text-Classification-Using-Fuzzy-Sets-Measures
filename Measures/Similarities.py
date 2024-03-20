import numpy as np

'''
	Similarity measures for (intuitionistic) fuzzy sets
	========================================================================
 	@author: nikolaos damianos
	© Visual Computing Group, HUMAIN-Lab, Eastern Macedonia and Thrace Institute of Technology
	01-Dec-2017 - Release
'''


def Similarities_():
    return ['dengChunSimilarity',
            'liangShiSimilarity',
            'parkKwunLimSimilarity',
            'mitchellSimilarity',
            'julHungLinSimilarity',
            'hungYangSimilarity',
            'yenSimilarity',
            'hwangYangSimilarity',
            'hungYangSimilarity2',
            'zhangFuSimilarity',
            'hungYangSimilarity3',
            'chenSimilarity',
            'hungYangDistance4',
            'hongKimSimilarity',
            'chenSimilarity2',
            'liuSimilarity',
            'iancuSimilarity',
            'songWangLeiXueSimilarity',
            'intarapaiboonSimilarity',
            'dengjiangFuSimilarity',
            'nguyenSimilarity',
            'chenChengLanSimilarity',
            'muthuKrishSimilarity']


#####################################################################################################################

def similarity(similarityMeasure, A, B, p=1, w=None, type=None, omegas=[0.5, 0.3, 0.2], a=1, b=0, c=0, lamda=1):
    if similarityMeasure == 'dengChunSimilarity':
        return __IFS_Sim01(A, B, p, w)  # Done
    elif similarityMeasure == 'liangShiSimilarity':
        return __IFS_Sim02(A, B, p=p, type=type, w=w, omegas=omegas)  # Done
    elif similarityMeasure == 'parkKwunLimSimilarity':
        return __IFS_Sim03(A, B, p, w)  # Done
    elif similarityMeasure == 'mitchellSimilarity':
        return __IFS_Sim04(A, B, p, w)  # Done
    elif similarityMeasure == 'julHungLinSimilarity':
        return __IFS_Sim05(A, B, p, w, type)  # Done
    elif similarityMeasure == 'hungYangSimilarity':
        return __IFS_Sim06(A, B, type, w)  # Done
    elif similarityMeasure == 'yenSimilarity':
        return __IFS_Sim07(A, B, w)  # Done
    elif similarityMeasure == 'hwangYangSimilarity':
        return __IFS_Sim08(A, B)  # Don
    elif similarityMeasure == 'hungYangSimilarity2':
        return __IFS_Sim09(A, B, type=type, a=a)  # Done
    elif similarityMeasure == 'zhangFuSimilarity':
        return __IFS_Sim10(A, B)
    elif similarityMeasure == 'hungYangSimilarity3':
        return __IFS_Sim11(A, B, type)  # Done
    elif similarityMeasure == 'chenSimilarity':
        return __IFS_Sim12(A, B, w)  # Done
    elif similarityMeasure == 'hungYangDistance4':
        return __IFS_Sim13(A, B, p, type)  # Done
    elif similarityMeasure == 'hongKimSimilarity':
        return __IFS_Sim14(A, B, w, a, b, c)  # Done
    elif similarityMeasure == 'chenSimilarity2':
        return __IFS_Sim15(A, B, w, a=1, b=0, c=0)
    elif similarityMeasure == 'liuSimilarity':
        return __IFS_Sim16(A, B, p, w, a=0.3, b=0.3, c=0.4)  # Done
    elif similarityMeasure == 'iancuSimilarity':
        return __IFS_Sim17(A, B, type, lamda)
    elif similarityMeasure == 'songWangLeiXueSimilarity':
        return __IFS_Sim18(A, B, w)
    elif similarityMeasure == 'intarapaiboonSimilarity':
        return __IFS_Sim19(A, B)
    elif similarityMeasure == 'dengjiangFuSimilarity':
        return __IFS_Sim20(A, B)
    elif similarityMeasure == 'nguyenSimilarity':
        return __IFS_Sim21(A, B)
    elif similarityMeasure == 'chenChengLanSimilarity':
        return __IFS_Sim22(A, B)
    elif similarityMeasure == 'muthuKrishSimilarity':
        return __IFS_Sim23(A, B)
    else:
        print(similarityMeasure)
        raise ValueError('Unknown similarity measure input.')


#####################################################################################################################

def __IFS_Sim01(A, B, p, w):
    '''
	IFS_SIM01: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by L. Dengfeng, C. Chuntian, from the related article 
	"New similarity measures of intuitionistic fuzzy sets and application to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		p: power parameter. Must be equal or higher than 1.
		w: weights for the computed similarity. If None then the non-weighted
		similarity is computed.
	OUTPUT:
		S: Similarity measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is not None:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')
        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')
        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if p < 1:
        raise ValueError('p parameter must be equal or higher than 1.')

    fA = (A_m + 1 - A_v) / 2.0
    fB = (B_m + 1 - B_v) / 2.0
    n = A.shape[0]
    if w is None:
        return 1 - 1 / (n ** (1 / p)) * (np.sum(np.absolute(fA - fB) ** p) ** (1 / p))
    else:
        return 1 - np.power(np.sum(w * np.power(np.absolute(fA - fB), p)), (1.0 / float(p)))


##############################################################################################################################################

def __IFS_Sim02(A, B, p, type, w=np.array([]), omegas=[0.5, 0.3, 0.2]):  # , varagrin):
    '''
	IFS_SIM02: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by Z. Liang, P. Shi, from the related article 
	"Similarity measures on intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed similarity: 'e', 's' or 'h'.
		p: power parameter. Must be equal or higher than 1.
		w: weights for the computed similarity. Used only for type = 'h'. If None
		then normalized weights are calculated.
		omegas: matrix containing values that belong to [0,1]. The sum of the 
		matrix must be equal to 1. Used only for type = 'h'.
	OUTPUT:
		S: Similarity measure.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    n = A.shape[0]

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if p < 1 and not isinstance(p, int):
        raise ValueError('p parameter must be an integer > 0')

    if type == 'h':
        if w is None:
            w = np.full(A.shape[0], 1 / float(A.shape[0]))
        else:
            w = np.array(w)
            if (w.shape[0] != A.shape[0] or w.ndim != 1):
                raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')
            elif np.any(w < 0) or np.any(w > 1):
                raise ValueError('Weight values must be 0 <= w <= 1.')
            elif w.sum() != 1:
                raise ValueError('Sum of weights must be equal to 1')

        omegas = np.array(omegas)

        if omegas.shape != (3,):
            raise ValueError('Omegas parameter must be a 3x1 matrix.')

    if (type == 'e'):
        mDif = np.absolute(A_m - B_m)
        vDif = np.absolute(A_v - B_v)

        D = np.power((np.sum(np.power(((mDif + vDif) / 2.0), p))), (1.0 / float(p)))
        return 1 - (1.0 / float(np.power(n, (1.0 / p)))) * D
    elif (type == 's'):
        mA = (A_m + 1 - A_v) / 2.0
        mB = (B_m + 1 - B_v) / 2.0
        mA1 = (A_m + mA) / 2.0
        mB1 = (B_m + mB) / 2.0
        mA2 = (mA + 1 - A_v) / 2.0
        mB2 = (mB + 1 - B_v) / 2.0
        fS1 = np.absolute(mA1 - mB1) / 2.0
        fS2 = np.absolute(mA2 - mB2) / 2.0
        D = np.power(np.sum(np.power(fS1 + fS2, p)), (1.0 / p))
        return 1 - 1.0 / float(np.power(n, (1.0 / p))) * D
    elif (type == 'h'):
        f1 = np.absolute(A_m - B_m) / 2.0 + np.absolute(A_v - B_v) / 2.0  # abs((A(:,2)-B(:,2)))/2
        mA = (A_m + 1 - A_v) / 2.0
        mB = (B_m + 1 - B_v) / 2.0
        f2 = np.absolute(mA - mB)
        iA = mA - A_m
        iB = mB - B_m
        f3 = np.maximum(iA, iB) - np.minimum(iA, iB)
        D = np.power(np.sum(w * (np.power(omegas[0] * f1 + omegas[1] * f2 + omegas[2] * f3, p))), (1.0 / p))
        return 1 - D

    else:
        raise ValueError('Distance type parameter must be e, s or h')


########################################################################################################################

def __IFS_Sim03(A, B, p, w):
    '''
	IFS_SIM03: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by A_v.H. Park, A_v.S. Park, Y.C. Kwun, K.M. Lim, from 
	the related article "New similarity measures on intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic  fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx3 matrices.')

    if w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    else:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if p < 1:
        raise ValueError('p parameter must be equal or higher than 1.')

    D = np.sum(
        w * np.power(np.absolute(A_m - B_m) / 2.0 + np.absolute(A_v - B_v) / 2.0 + np.absolute(A_p - B_p) / 2.0, p))

    return 1 - (np.power(D, (1.0 / p)))


########################################################################################################################

def __IFS_Sim04(A, B, p, w):
    '''
	IFS_SIM04: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by H.B. Mitchell, from the related article 
	"On the Dengfeng-Chuntian similarity measure and its application to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		p: must be a positive integer.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    else:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if p < 1 and not isinstance(p, int):
        raise ValueError('p parameter must be integer')

    D_m = 1 - np.power(np.sum(w * np.power(np.absolute(A_m - B_m), p)), (1.0 / p))
    D_f = 1 - np.power(np.sum(w * np.power(np.absolute(A_v - B_v), p)), (1.0 / p))

    return (D_m + D_f) / 2.0


########################################################################################################################

def __IFS_Sim05(A, B, p, w, type):
    '''
	IFS_SIM05: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by P. A_vulian, K.C. Hung, S.A_v. Lin, from the related article 
	"On the Mitchell similarity measure and its application to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		p: must be a positive integer.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
		type: Type of computed similarity: 1 or 2.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    else:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if p < 1 and not isinstance(p, int):
        raise ValueError('p parameter must be integer')

    D_m = np.sum(w * np.power(np.absolute(A_m - B_m), p))
    D_f = np.sum(w * np.power(np.absolute(A_v - B_v), p))
    # to np.sum tha prepei na epistrepsei noumero, i an epistrefei pinaka (den tha eprepe) to kanei numpy array etsi k alliws
    # D_m = np.array(D_m)
    # D_f = np.array(D_f)

    if (type == 1):
        return 1 - np.power((D_m * 0, 6), (1.0 / p)) - np.power((D_f * 0.6), (1.0 / p))
    else:
        return 1 - np.power(D_m, (1.0 / p)) - np.power(D_f, (1.0 / p))


########################################################################################################################

def __IFS_Sim06(A, B, type, w):
    '''
	IFS_SIM06: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by W.L. Hung, M.S. Yang, from the related article 
	"Similarity measures of intuitionistic fuzzy sets based on Hausdorff similarity"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		type: Type of computed similarity: 'l', 'e' or 'c'
		w: weights for the computed similarity. If None then the non-weighted
		similarity is computed.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is not None:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if w is None:
        D = np.sum(np.maximum(np.absolute(A_m - B_m), np.absolute(A_v - B_v))) / float(len(A))
    else:
        D = np.sum(w * np.maximum(np.absolute(A_m - B_m), np.absolute(A_v - B_v)))

    if type == 'l':
        return 1 - D
    elif type == 'e':
        tp = np.exp(-1)
        return (np.exp(-D) - tp) / float(1 - tp)
    else:
        return (1 - D) / float(1 + D)


########################################################################################################################

def __IFS_Sim07(A, B, w):
    '''
	IFS_SIM07: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by A_v. Yen, from the related article 
	"Cosine similarity measures for intuitionistic fuzzy sets and their applications"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    else:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    num = A_m * B_m + A_v * B_v
    denum = np.sqrt(np.power(A_m, 2.0) + np.power(A_v, 2.0)) * np.sqrt(np.power(B_m, 2.0) + np.power(B_v, 2.0))

    if w is None:
        return 1 / len(A) * sum(num / denum)
    else:
        return 1 / len(A) * sum(w * num / denum)


########################################################################################################################

def __IFS_Sim08(A, B):
    '''
	IFS_SIM08: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by C.M.  Hwang and M.S. Yang, from the related article 
	"Modified cosine similarity measure between intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])  # kai num1
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    num = A_m * B_m + A_v * B_v
    denum = np.sqrt(np.power(A_m, 2.0) + np.power(A_v, 2.0)) * np.sqrt(np.power(B_m, 2.0) + np.power(B_v, 2.0))

    denum[denum == 0] = 1
    num[denum == 0] = 0

    C1 = 1 / len(A) * sum(num / denum)

    fA = (1 + A_m - A_v) / 2.0
    fB = (1 + B_m - B_v) / 2.0

    num = fA * fB + A_m * B_m
    denum = np.sqrt(fA ** 2 + A_v ** 2) * np.sqrt(fB ** 2 + B_v ** 2)

    C2 = 1.0 / len(A) * sum(num / np.array(denum).astype("float"))

    num = (1 - A_m) * (1 - B_m) + (1 - A_v) * (1 - B_v)
    denum = np.sqrt(np.square((1 - A_m)) + np.square((1 - A_v))) * np.sqrt(np.square((1 - B_m)) + np.square((1 - B_v)))

    C3 = 1 / len(A) * sum(num / np.array(denum).astype("float"))

    return (C1 + C2 + C3) / 3.0


########################################################################################################################

def __IFS_Sim09(A, B, type, a=1):
    '''
	IFS_SIM09: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by W.L. Hung, M.S. Yang, from the related article 
	"On the A_v-divergence of intuitionistic fuzzy sets with its applications to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree
		a: case of divergence measure. Positive integer.
		type: Type of computed similarity: 'l', 'e' or 'c'.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    n = A.shape[0]
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if a < 1 and not isinstance(a, int):
        raise ValueError('a parameter must be integer')

    if a == 1:
        U_a = np.log(2.0)

        part1 = (A_m + B_m) / 2
        part1 = np.where(part1 > 0, np.log(part1), part1)
        part2 = A_m
        part2 = np.where(part2 > 0, np.log(part2), part2)
        part3 = B_m
        part3 = np.where(part3 > 0, np.log(part3), part3)
        LABm = (A_m + B_m) * part1 - A_m * part2 - B_m * part3

        part1 = (A_p + B_p) / 2
        part1 = np.where(part1 > 0, np.log(part1), part1)
        part2 = A_p
        part2 = np.where(part2 > 0, np.log(part2), part2)
        part3 = B_p
        part3 = np.where(part3 > 0, np.log(part3), part3)

        LABp = (A_p + B_p) * part1 - A_p * part2 - B_p * part3

        part1 = (A_v + B_v) / 2
        part1 = np.where(part1 > 0, np.log(part1), part1)
        part2 = A_v
        part2 = np.where(part2 > 0, np.log(part2), part2)
        part3 = B_v
        part3 = np.where(part3 > 0, np.log(part3), part3)

        LABv = (A_v + B_v) * part1 - A_v * part2 - B_v * part3

        J = np.sum(-0.5 * np.sum(LABm + LABv + LABp))
    else:
        U_a = 1.0 / (float(a) - 1.0) * (1 - 1 / float(2.0 ** (a - 1.0)))
        TmAB = (np.power(((A_m + B_m) / 2.0), a)) - 0.5 * (np.power(A_m, a) + np.power(B_m, a))
        TvAB = (np.power(((A_v + B_v) / 2.0), a)) - 0.5 * (np.power(A_v, a) + np.power(B_v, a))
        TpAB = (np.power(((A_p + B_p) / 2.0), a)) - 0.5 * (np.power(A_p, a) + np.power(B_p, a))

        # A_v = np.sum(1.0/(a-1.0) * TAB)
        J = np.sum(-1 / (a - 1) * (TmAB + TvAB + TpAB))

    J_a = J / float(n)

    if (type == 'l'):
        return (U_a - J_a) / float(U_a)
    elif (type == 'e'):
        return (np.exp(-J_a) - np.exp(-U_a)) / float(1 - np.exp(-U_a))
    elif (type == 'c'):
        np.seterr(all='ignore')
        return (U_a - J_a) / np.array((1 + J_a) * U_a).astype("float")


########################################################################################################################


def __IFS_Sim10(A, B):
    '''
	IFS_SIM10: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by C. Zhang, H. Fu, from the related article 
	"Similarity measures on three kinds of fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    n = A.shape[0]
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx3 matrices.')

    dA = (A_m + A_p) * A_m
    aA = (A_v + A_p) * A_v

    dB = (B_m + B_p) * B_m
    aB = (B_v + B_p) * B_v

    D = np.sum(np.absolute(dA - dB) + np.absolute(aA - aB))

    return 1.0 - D / (2.0 * n)


########################################################################################################################

def __IFS_Sim11(A, B, type):
    '''
	IFS_SIM11: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by W.L. Hung and M.S. Yang, from the related article 
	"On similarity measures between intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D array containing in the first dimension the membership values
		and in the second the non-membership values.
		type: Type of computed similarity: 'w1', 'w2', 'pk1', 'pk2', 'pk3', 'new1'
		or 'new2'.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if type == 'w1':
        num = np.minimum(A_m, B_m) + np.minimum(A_v, B_v)
        denom = np.maximum(A_m, B_m) + np.maximum(A_v, B_v)
        D = np.sum(np.divide(num, denom.astype(float)))
        return D / float(len(A))
    elif type == 'w2':
        n = A.shape[0]
        ptr2 = 0.5 * np.absolute(A_m - B_m) + np.absolute(A_v - B_v)
        D = np.sum(np.subtract(1, ptr2))
        return D / float(n)
    elif type == 'pk1':
        num = np.minimum(A_m, B_m) + np.minimum(A_v, B_v)
        denom = np.maximum(A_m, B_m) + np.maximum(A_v, B_v)
        D1 = np.sum(num)
        D2 = np.sum(denom)
        return D1 / float(D2)
    elif type == 'pk2':
        dif_m = np.absolute(A_m - B_m)
        dif_v = np.absolute(A_v - B_v)
        return 1 - 0.5 * (dif_m.max() + dif_v.max())
    elif type == 'pk3':
        num = np.absolute(A_m - B_m) + np.absolute(A_v - B_v)
        denom = np.absolute(A_m + B_m) + np.absolute(A_v + B_v)
        D1 = np.sum(num)
        D2 = np.sum(denom)
        # na to 3ana dw
        return 1 - (D1 / float(D2))
    elif type == 'new1':
        num = 1 - np.exp(-0.5 * sum(np.absolute(A_m - B_m) + np.absolute(A_v - B_v)))
        denum = 1 - np.exp(-len(A))
        return 1 - num / denum
    elif type == 'new2':
        num = 1 - np.exp(
            -0.5 * np.sum(np.absolute(np.sqrt(A_m) - np.sqrt(B_m)) + np.absolute(np.sqrt(A_v) - np.sqrt(B_v))))
        denum = 1 - np.exp(-len(A))
        return 1 - num / denum
    else:
        raise ValueError('Distance type parameter must be w1, w2, pk1, pk2, pk3, new1 or  new2 .')


########################################################################################################################

def __IFS_Sim12(A, B, w):
    '''
	IFS_SIM12: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by S.M. Chen, from the related article 
	"Measures of similarity between vague sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    else:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')
        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

    sA = A_m - A_v
    sB = B_m - B_v
    D = np.sum(w * (np.subtract(1, np.absolute(sA - sB) / 2.0)))

    return D / float(np.sum(w))


########################################################################################################################

def __IFS_Sim13(A, B, p, type):
    '''
	IFS_SIM13: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by W.L. Hung and M.S. Yang, from the related article 
	"Similarity measures of intuitionistic fuzzy sets based on Lp metric"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		p: must be a positive integer.
		type: Type of computed similarity: 'l', 'e' or 'c'
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if p < 1 and not isinstance(p, int):
        raise ValueError('p parameter must be integer')

    D = np.sum(np.power((np.absolute(np.power(A_m - B_m, p)) + np.power(np.absolute(A_v - B_v), p)), (1.0 / p)))

    D = D / float(len(A))
    f1 = 2 ** (1.0 / float(p))

    if type == 'l':
        return (f1 - D) / float(f1)
    elif type == 'e':
        return (np.exp(-D) - np.exp(-f1)) / float(1 - np.exp(-f1))
    elif type == 'c':
        return (f1 - D) / float(f1 * (1.0 + D))


########################################################################################################################

def __IFS_Sim14(A, B, w, a=1, b=0, c=0):
    '''
	IFS_SIM14: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by D.H. Hong and C.Kim, from the related article 
	"A note on similarity measures between vague sets and between elements"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
		a, b, c: Positive integers.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    n = A.shape[0]

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if w is not None:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if a < 0 or b < 0 or c < 0:  # or a > 1 or b > 1 or c > 1:
        raise ValueError('Parameters ''a'', ''b'' and ''c'' must be equal or higher than 0')

    if w is None:
        D = np.sum(np.subtract(1, (np.absolute(A_m - B_m) + np.absolute(A_v - B_v)) / 2.0))
        return D / float(n)
    else:
        denom = a + b + c

        num = (a * np.absolute(A_m - B_m)) + (b * np.absolute(A_v - B_v)) + (c * np.absolute(B_m + B_v - A_m - A_v))

        D = np.sum(w * (1 - num / float(denom)))  # np.sum(np.multiply(w,np.divide(np.subtract(1,num),float(denom))))

        return D


########################################################################################################################

def __IFS_Sim15(A, B, w, a=1, b=0, c=0):
    '''
	IFS_SIM15: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarity proposed by S.M. Chen, from the related article 
	"Similarity measure between vague sets and between elements"
	
	INPUTS:
		A, B: 2-D array containing in the first row the membership values and
		in the second the non-membership values.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
		A,B,b,c: Positive integers.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if (a + b + c) != 1:
        raise ValueError('Sum of parameters ''a'', ''b'' and ''c'' must be equal to 1.')

    elif a < c or c < 0 or 0 < b:
        raise ValueError('a, b and c parameters must meet this condition: a >= c >= 0 >= b.')

    elif (a < 0 or a > 1) or (b < 0 or b > 1) or (c < 0 or c > 1):
        raise ValueError('Parameters a, b and c must be 0 <= a, b, c <= 1.')

    if w is None:
        w = np.full(A.shape[0], 1 / float(A.shape[0]))
    else:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    denom = a - b
    num = (a * (A_m - B_m)) + (b * (A_v - B_v)) - (c * (A_m - B_m + A_v - B_v))
    D = np.sum(w * (1 - np.absolute(num) / float(denom)))
    D = np.array(D)

    return D / float(np.sum(w))


########################################################################################################################

def __IFS_Sim16(A, B, p, w, a=1, b=0, c=0):
    '''
	IFS_SIM16: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by H.W. LA_m, from the related article 
	"New similarity measures between intuitionistic fuzzy sets and between elements"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		p: must be a positive integer.
		w: weights for the computed similarity. If None then normalized weights are
		calculated
		a,b,c: Positive integers. Must a + b + c = 1.
	OUTPUT:
		S: Similarity between the intuitionistic fuzzy set A and the intuitionistic 
		fuzzy set of the ideally segmented image.
    '''

    n = A.shape[0]

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx3 matrices.')

    if w is not None:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if p < 1 and not isinstance(p, int):
        raise ValueError('p parameter must be integer')

    if w is None:
        D = np.sum((np.power(np.absolute(A_m - B_m), p)) + (np.power(np.absolute(A_v - B_v), p)) + (
            np.power(np.absolute(A_p - B_p), p)))
        return 1 - (D / float(2.0 * n)) ** (1.0 / p)
    else:
        prts = (a * np.power(np.absolute(A_m - B_m), p)) + (b * np.power(np.absolute(A_v - B_v), p)) + (
                c * np.power(np.absolute(A_p - B_p), p))
        D = np.sum(w * prts)
        return 1 - D ** (1.0 / p)


########################################################################################################################

def tOperator(a, b, lamda):
    a = np.array(a)
    b = np.array(b)
    if (lamda == np.inf):
        return np.maximum(0, a + b - 1)
    elif (lamda == 0):
        return np.minimum(a, b)
    elif (lamda == 1):
        return a * b
    else:
        log_num = (np.power(lamda, np.subtract(1, a)) - 1) * (np.power(lamda, np.subtract(1, b)) - 1)
        log_denum = lamda - 1
        log_denum = np.array(log_denum)
        return np.log(1.0 + log_num / log_denum.astype(float))


########################################################################################################################

def __IFS_Sim17(A, B, type, lamda=1):
    '''
	IFS_SIM17: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by Ion Iancu, from the related article 
	"Intuitionistic fuzzy similarity measures based on Frank t-norms family"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		type: Type of computed similarity: 1-20.
		lambda: Frank family of t-operator parameter. Different cased of input:
		0, 1, Inf and other numbers.
	OUTPUT:
		S: Similarity measure.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    n = A.shape[0]
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    if (type == 1):  # S11
        nominator = n + np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n + np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / np.array(denom).astype("float")
    elif (type == 2):  # S11c
        nominator = n - np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n - np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / np.array(denom).astype("float")
    elif (type == 3):  # S19
        nominator = n + np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / (2.0 * n)
    elif (type == 4):  # S19c
        nominator = n - np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nominator / (2.0 * n)
    elif (type == 5):  # S1
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_v - B_v)
        denom = n + np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nom / np.array(denom).astype("float")
    elif (type == 6):  # S1c
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_m - B_m)
        denom = n - np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        return nom / np.array(denom).astype("float")
    elif (type == 7):  # S5
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_v - B_v)
        denom = n + np.sum(A_m + B_m - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / np.array(denom).astype("float")
    elif (type == 8):  # S5c
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_m - B_m)
        denom = n + np.sum(A_v + B_v - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / np.array(np.sum(denom)).astype("float")
    elif (type == 9):  # S14
        nom = n + np.minimum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n + np.sum(A_m + B_m - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / float(denom)
    elif (type == 10):  # S14c
        nom = n - np.maximum(np.sum(A_m - A_v), np.sum(B_m - B_v))
        denom = n + np.sum(A_v + B_v - tOperator(A_m, B_m, lamda) - tOperator(A_v, B_v, lamda))
        return nom / float(denom)
    elif (type == 11):  # S18
        nom = n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_v - B_v)
        denom = 2.0 * n
        return nom / denom
    elif (type == 12):  # S18c
        nom = n - np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda) - A_m - B_m)
        denom = 2.0 * n
        return nom / denom
    elif (type == 13):  # S2
        nom = 2.0 * n + np.sum(
            2.0 * tOperator(A_m, B_m, lamda) + 2.0 * tOperator(A_v, B_v, lamda) - A_m - B_m - A_v - B_v)
        denom = 2.0 * n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda)) - np.minimum(
            np.sum(A_m + B_v), np.sum(B_m + A_v))
        return nom / denom
    elif (type == 14):  # S6
        nom = 2.0 * n + np.sum(
            2.0 * tOperator(A_m, B_m, lamda) + 2.0 * tOperator(A_v, B_v, lamda) - A_m - B_m - A_v - B_v)
        denom = 2.0 * n
        return nom / denom
    elif (type == 15):  # S13
        nom = np.minimum(np.sum(A_m + B_v), np.sum(B_m + A_v)) - np.sum(
            tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda))
        denom = np.sum(A_m + B_m + A_v + B_v - 2.0 * tOperator(A_m, B_m, lamda) - 2 * tOperator(A_v, B_v, lamda))
        return nom / denom
    elif (type == 16):  # S15
        nom = 2.0 * n + np.sum(tOperator(A_m, B_m, lamda) + tOperator(A_v, B_v, lamda)) - np.maximum(np.sum(A_m + B_v),
                                                                                                     np.sum(B_m + A_v))
        denom = 2.0 * n
        return nom / denom
    elif (type == 17):  # S2'
        nom = np.sum(A_m + B_m + A_v + B_v - 2.0 * tOperator(A_m, B_v, lamda) - 2.0 * tOperator(B_m, A_v, lamda))
        denom = n - np.sum(tOperator(A_m, B_v, lamda) + tOperator(B_m, A_v, lamda)) + np.maximum(np.sum(A_m + B_v),
                                                                                                 np.sum(B_m + A_v))
        return nom / denom.astype(float)
    elif (type == 18):  # S6'
        nom = np.sum(A_m + B_m + A_v + B_v - 2.0 * tOperator(A_m, B_v, lamda) - 2.0 * tOperator(B_m, A_v, lamda))
        denom = 2.0 * n
        return nom / denom
    elif (type == 19):  # S13'
        nom = n + np.sum(tOperator(A_m, B_v, lamda) + tOperator(B_m, A_v, lamda)) - np.maximum(np.sum(A_m + B_v),
                                                                                               np.sum(B_m + A_v))
        denom = 2.0 * n + np.sum(
            2.0 * tOperator(A_m, B_v, lamda) + 2.0 * tOperator(B_m, A_v, lamda) - A_m - B_m - A_v - B_v)
        return nom / denom
    elif (type == 20):  # S15'
        nom = n + np.minimum(np.sum(A_m + B_v), np.sum(B_m + A_v)) - np.sum(
            tOperator(A_m, B_v, lamda) + tOperator(B_m, A_v, lamda))
        denom = 2.0 * n
        return nom / denom
    else:
        raise ValueError("Parameter type must be >= 1 and <= 20.")


########################################################################################################################

def __IFS_Sim18(A, B, w):
    '''
	IFS_SIM18: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by Yafei Song, Xiaodan Wang, Lei Lei, AiA_vun Xue, from the related article 
	"A novel similarity measure on intuitionistic fuzzy sets with its applications"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
		w: weights for the computed similarity. If None then the non-weighted similarity
		is calculated.
	OUTPUT:
		S: Similarity measure.
    '''

    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    n = A.shape[0]
    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if w is not None:
        w = np.array(w)
        if (w.shape[0] != A.shape[0] or w.ndim != 1):
            raise ValueError('Weight parameter dimensions must be 1-D Nx1 matrix.')

        elif np.any(w < 0) or np.any(w > 1):
            raise ValueError('Weight values must be 0 <= w <= 1.')

        elif w.sum() != 1:
            raise ValueError('Sum of weights must be equal to 1')

    if w is None:
        ff = 1.0 / (2.0 * n) * np.sum(
            np.sqrt(A_m * B_m) + np.multiply(2.0, np.sqrt(A_v, B_v)) + np.sqrt(A_p * B_p) + np.sqrt(
                (1 - A_v) * (1 - B_v)))
        return ff
    else:
        lll = np.sum(
            w * (np.sqrt(A_m * B_m) + 2.0 * np.sqrt(A_v * B_v) + np.sqrt(A_p, B_p) + np.sqrt((1 - A_v) * (1 - B_v))))
        ff = 1.0 / 2.0 * lll
        return ff


########################################################################################################################

def __IFS_Sim19(A, B):
    '''
	IFS_SIM19: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	similaritys proposed by Peerasak Intarapaiboon, from the related article 
	"A hierarchy-based similarity measure for intuitionistic fuzzy sets"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
	OUTPUT:
		S: Similarity measure.
    '''
    n = A.shape[0]
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    ff = 1 - (1.0 / (2.0 / n) * np.sum(np.absolute(A_m - B_m) + np.absolute(A_v - B_v)))
    return ff


########################################################################################################################

def __IFS_Sim20(A, B):
    '''
	IFS_SIM20: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by Guannan Deng, Yanli A_viang, A_vingchao Fu, from the related article 
	"Monotonic similarity measures between intuitionistic fuzzy sets and their relationship with 
	entropy and inclusion measure"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values,
		in the second the non-membership values and in the third the hesitation
		degree.
	OUTPUT:
		S: Similarity measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    gg = float(np.sum(np.minimum(A_m, B_m) + np.minimum(A_v, B_v))) / np.sum(
        np.maximum(A_m, B_m) + np.maximum(A_v, B_v))
    return gg


########################################################################################################################

def __IFS_Sim21(A, B):
    '''
	IFS_SIM22: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Hoang Nguyen, from the related article 
	"A novel similarity/dissimilarity measure for intuitionistic fuzzy sets and its application
	in pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		S: Similarity measure.
    '''
    n = A.shape[0]
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx3 matrices.')

    Ka = 1.0 / (n * np.sqrt(2)) * np.sum(np.sqrt(np.power(A_m, 2.0) + np.power(A_v, 2.0) + (1 - np.power(A_p, 2.0))))
    Kb = 1.0 / (n * np.sqrt(2)) * np.sum(np.sqrt(np.power(B_m, 2.0) + np.power(B_v, 2.0) + (1 - np.power(B_p, 2.0))))

    return 1 - np.absolute(Ka - Kb)


########################################################################################################################

def __IFS_Sim22(A, B):
    '''
	IFS_SIM22: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Distances proposed by Shyi-Ming Chen, Shou-HsA_mng Cheng,Tzu-Chun Lan, from the related article 
	"A novel similarity measure between intuitionistic fuzzy sets based on the centroid 
	points of transformed fuzzy numbers with applications to pattern recognition"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		S: Similarity measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])
    A_p = np.array(A[:, 2])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])
    B_p = np.array(B[:, 2])

    if A.shape[1] != 3 or not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx3 matrices.')

    term1 = np.absolute(2.0 * (A_m - B_m) - (A_v - B_v)) / 3.0 * (1 - ((A_p + B_p) / 2.0))
    term2 = np.absolute(2.0 * (A_v - B_v) - (A_m - B_m)) / 3.0 * ((A_p + B_p) / 2.0)

    return 1 - np.sum(term1) - np.sum(term2)


########################################################################################################################

def __IFS_Sim23(A, B):
    '''
	IFS_SIM23: Calculates similarity between the intuitionistic fuzzy 
	sets A and B.
	Similarities proposed by P. MuthukumarA,B, G. Sai Sundara KrishnanbA,B, from the related article 
	"A similarity measure of intuitionistic fuzzy soft sets and itsapplication in medical diagnosis"
	
	INPUTS:
		A, B: 2-D matrix containing in the first row the membership values and
		in the second the non-membership values.
	OUTPUT:
		S: Similarity measure.
    '''
    A_m = np.array(A[:, 0])
    A_v = np.array(A[:, 1])

    B_m = np.array(B[:, 0])
    B_v = np.array(B[:, 1])

    if (A.shape[1] != 2 or B.shape[1] != 2) and not (A.shape == B.shape):
        raise ValueError('A and B parameters must be 2-D Nx2 matrices.')

    nom = np.sum(A_m * B_m + A_v * B_v)
    denom = np.sum(
        np.maximum(np.power(A_m, 2.0), np.power(B_m, 2.0)) + np.maximum(np.power(A_v, 2.0), np.power(B_v, 2.0)))
    return nom / float(denom)
