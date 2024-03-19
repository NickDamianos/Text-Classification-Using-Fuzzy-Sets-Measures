# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:43:40 2020

@author: nikolaos damianos
"""



from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import numpy as np
#import Distances

import sys
sys.path.append("..")


from Measures.Distances import  Distances_ , distance
from Measures.Similarities import  Similarities_ , similarity

'''
from Distances import distance, Distances
from Similarities import similarity, Similarities
'''


def dist_param():
    #returns the names and parameters from distances
    # in dictionary
    distances = Distances_()
    params = {}
    for dist in distances:
        if not dist == 'vlachSergDistance':
            if dist == 'wangXinDistance':
                params[dist] = [1, 2]
            else:
                params[dist] = ['H', 'E', 'nH', 'nE']
        else:
            params[dist] = '-'

    return params


def sim_param():
    #returns the names and parameters from similarities 
    # in dictionary
    similarities = Similarities_()
    params = {}

    for sim in similarities:
        if sim in [similarities[1], similarities[4], similarities[5], similarities[8], similarities[10],
                   similarities[12], similarities[16]]:
            if sim == similarities[1]:
                params[sim] = ['e', 's', 'h']
            elif sim == similarities[4]:
                params[sim] = [1, 2]
            elif sim in [similarities[5], similarities[8], similarities[12]]:
                params[sim] = ['l', 'e', 'c']
            elif sim == similarities[10]:
                params[sim] = ['w1', 'w2', 'pk1', 'pk2', 'pk3', 'new1', 'new2']
            else:
                params[sim] = [i for i in range(1, 21)]
        else:
            params[sim] = '-'

    return params




def calculate_membership(data, rj1, rj2, Xj=None, sj=None):
    # calculate the mean of each word
    if Xj is None:
        Xj = np.mean(data, axis=0, dtype=np.float64)
    # calculate the standard deviation of each word
    if sj is None:
        sj = np.std(data, axis=0, dtype=np.float32, ddof=1)

    # calculate membership values of each word for each document
    z = (data - Xj) / sj
    z[np.isnan(z)] = 0
    m = rj1 / (1 + np.exp(-z))
    v = rj2 / (1 + np.exp(z))
    p = 1 - m - v

    return m, v, p, Xj, sj


def fuzzy_train(data, targets, rj1, rj2):
    """
        Returns the memberships of each word , the mean of each word and the standard deviation 
    """
    docs, words = np.array(data).shape
    # print(docs, words)
    # print ("\tCalculating (non)membership values...")
    m, v, p, Xj, sj = calculate_membership(data, rj1, rj2)

    # print ("\tCalculating class patterns...")
    numberOfClasses = len(np.unique(targets))

    # Pk[0] - > first class
    # Pk[2][0] -> values (membership/etc) of the first word for the third class
    # Pk[3][0][1] -> non-membership value of the first word for the forth class
    Pk = np.zeros((numberOfClasses, words, 3), dtype=np.float64)  # 3 for m v p
    classes = np.unique(targets)
    docs = np.asarray(range(docs))
    for c in range(len(classes)):
        same_class_docs = docs[targets == classes[c]]
        if len(same_class_docs) == 1:
            same_class_docs = same_class_docs[0]

        Pk[c, :, 0] = np.average(m[same_class_docs, :], axis=0)
        Pk[c, :, 1] = np.average(v[same_class_docs, :], axis=0)
        Pk[c, :, 2] = np.average(p[same_class_docs, :], axis=0)

    return Pk, Xj, sj


def fuzzy_test(test_data, Xj, sj, rj1, rj2):
    """
        Returns the memberships of each word 
    """
    
    #print("Testing...")

    print("\tCalculating (non)membership values...")
    docs, words = test_data.shape

    A = np.zeros((docs, words, 3), dtype=np.float64)
    # calculate the membership values of each word for each document
    for doc in range(docs):
        A[doc, :, 0], A[doc, :, 1], A[doc, :, 2], _, _ = calculate_membership(test_data[doc, :], rj1, rj2, Xj, sj)

    return A



def cross_validate(rj1, rj2, train_test):
    """
    returns 2 lists with ((name,parameter , mean of the smallest distance , mean of Doconf , Accuracy, Precision, Recall,
    Roc )) of the distancies and similarities 
    """
    X_train, y_train, X_test, y_test = train_test
    class_patterns, Xj, sj = fuzzy_train(X_train, y_train, rj1, rj2)

    # count_vect_test = CountVectorizer(min_df=0, vocabulary=train_vocabulary)

    X_test = np.asarray(X_test)

    # # calculate membership/etc values for each word in each document
    IFS = fuzzy_test(X_test, Xj, sj, rj1, rj2)

    # classify the new documents

    dist_result = fuzzy_classify_all_Dist(class_patterns,IFS,y_test)
    sim_result = fuzzy_classify_all_Similarities(class_patterns, IFS, targets=y_test)

    return np.asarray(dist_result), np.asarray(sim_result)




def fuzzy_classify_all_Dist(Pk, IFS, targets=None):
    '''
    @parameters Pk = classes , IFS = ifs of documents ,  targets = target classes
    
    returns a list with distances (name,parameter , mean of the smallest distance , mean of Doconf , Accuracy, Precision, Recall,
    Roc ) 
    '''
    classes = Pk
    classes.flags.writeable = False

    print("\tClassifying... (Distancies)")
    distances = []

    doc_results = []
    doc_mins = []

    dis = Distances_()
    params = dist_param()

    i = 0
    for d in dis:#for every distance
        # print('Dist ' + str(i) + " / " + str(cnt))
        i += 1
        par = params[d]

        for p in par:#for every parameter 
            Doconf = []
            doc_cnt = 0
            for doc_words in IFS:  # for each document
                # print('Dist ' + str(i) + " / " + str(cnt), '  ', p, ' , ', 'Doc ' + str(doc_cnt) + '/' + str(len(IFS)-1))
                results = []
            
                for class_ in Pk:  # for each class
                    # calculate the distance between the class and the current document doc_words
                    if p is None:
                        p = 1

                    ifsDist = distance(d, doc_words, class_, type=p)
                    results.append(ifsDist)  # append the distance value

                # get the class that has the mimimum distance with the current document
                doc_results.append(np.argmin(results))  # np.argmin(results))
                doc_mins.append(np.min(results))

                Doconf.append(np.sum(np.absolute(np.min(results) - results)))
                doc_cnt += 1

                # Doconf = []

            # print(d + ' ' + str(p) + ' Acuracy : ', accuracy_score(targets,doc_results))
            # f = error_calc1(doc_results, targets)##classes[mins])#, List=True)

            Accuracy = accuracy_score(targets, doc_results)
            Precision = precision_score(targets, doc_results, average='macro')
            Recall = recall_score(targets, doc_results, average='macro')
            fpr, tpr, thresholds = roc_curve(targets, doc_results,
                                             pos_label=int(len(np.unique(targets)) / 2))  # ROC curve

            # fpr : Increasing false positive rates
            # tpr : Increasing true positive rates
            # thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
            Roc = auc(fpr, tpr)  # auc score

            Doconf = np.absolute(Doconf - np.mean(Doconf)) / np.std(Doconf)
            distances.append([d, p, np.mean(doc_mins), np.mean(Doconf), Accuracy, Precision, Recall,
                              Roc])  # roc_auc_score(targets,doc_results)])
            doc_results = []
            doc_mins = []

    return distances


def fuzzy_classify_all_Similarities(Pk, IFS, lamba_=None, targets=None):
    '''
    @parameters Pk = classes , IFS = ifs of documents ,  targets = target classes , lamba_= lamba
    returns a list with Similarities (name,parameter , mean of the biggest Similarities , mean of Doconf , Accuracy, Precision, Recall,
    Roc ) 
    '''
    
    classes = Pk
    classes.flags.writeable = False
    print("\tClassifying... (Similarities)")

    similarities = []

    doc_results = []
    doc_maxes = [] 

    params = sim_param()
    sims = Similarities_()

    omegas = [0.5, 0.3, 0.2]

    lamba = lamba_
    if lamba_ is None:
        lamba = 1

    i = 0

    for s in sims:
        i += 1
        par = params[s]

        for p in par:
            doc_cnt = 0
            Doconf = []
            for doc_words in IFS:

                results = []

                for class_ in Pk:

                    if p is None:
                        p = 1

                    ifsSim = similarity(s, doc_words, class_, type=p, omegas=omegas, lamda=lamba)
                    results.append(ifsSim)

                doc_results.append(np.argmax(results))  # np.argmin(results))
                doc_maxes.append(np.max(results))

                Doconf.append(np.sum(np.absolute(np.max(results) - results)))
                doc_cnt += 1

            # print(s + ' '+ str(p) + ' Acuracy :', accuracy_score(targets,doc_results))

            Accuracy = accuracy_score(targets, doc_results)
            Precision = precision_score(targets, doc_results, average='micro')
            Recall = recall_score(targets, doc_results, average='micro')
            fpr, tpr, thresholds = roc_curve(targets, doc_results,
                                             pos_label=int(len(np.unique(targets)) / 2))  # ROC curve
            # fpr : Increasing false positive rates
            # tpr : Increasing true positive rates
            # thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
            try:
                Roc = auc(tpr, fpr)  # auc score
            except:
                Roc = 0.01

            Doconf = np.absolute(Doconf - np.mean(Doconf)) / np.std(Doconf)

            similarities.append([s, p, np.mean(doc_maxes), np.mean(Doconf), Accuracy, Precision, Recall, Roc])
            doc_results = []
            doc_maxes = []

    return similarities


def classify_distance(Pk, IFS, dist,param ,targets=None):
    '''
    @parameters Pk = classes , IFS = ifs of documents ,  targets = target classes , dist = name of distance , 
    param = parameter of distance
    
    returns  (name,parameter , mean of the smallest distance , mean of Doconf , Accuracy, Precision, Recall,
    Roc ) of the distance given
    '''
    
    
    classes = Pk
    classes.flags.writeable = False

    print("\tClassifying... (Distance : " + dist +" _ " + param + " )")
    

    doc_results = []
    doc_mins = []
    
    Doconf = []
    
    for doc_words in IFS:  # for each document
                
        results = []

        for class_ in Pk:  # for each class
            # calculate the distance between the class and the current document doc_words
            

            ifsDist = distance(dist, doc_words, class_, type=param)
            results.append(ifsDist)  # append the distance value

            # get the class that has the mimimum distance with the current document
            doc_results.append(np.argmin(results))  # np.argmin(results))
            doc_mins.append(np.min(results))

            Doconf.append(np.sum(np.absolute(np.min(results) - results)))
            
    Accuracy = accuracy_score(targets, doc_results)
    Precision = precision_score(targets, doc_results, average='macro')
    Recall = recall_score(targets, doc_results, average='macro')
    fpr, tpr, thresholds = roc_curve(targets, doc_results,
                                        pos_label=int(len(np.unique(targets)) / 2))  # ROC curve

    # fpr : Increasing false positive rates
    # tpr : Increasing true positive rates
    # thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
    try:
        Roc = auc(tpr, fpr)  # auc score
    except:
        Roc = 0.01

    Doconf = np.absolute(Doconf - np.mean(Doconf)) / np.std(Doconf)
    Distance= [dist, param, np.mean(doc_mins), np.mean(Doconf), Accuracy, Precision, Recall,
                              Roc]  # roc_auc_score(targets,doc_results)])
        
    return Distance     
    
    

def classify_Similarity(Pk, IFS,sim,param, lamba_=None, targets=None,omegas = [0.5, 0.3, 0.2]):
    '''
    @parameters Pk = classes , IFS = ifs of documents ,  targets = target classes , lamba_= lamba , omegas = omegas
    sim = name of similarity ,param = parameter of similarity
    
    returns  (name,parameter , mean of the biggest Similarities , mean of Doconf , Accuracy, Precision, Recall,
    Roc ) of the similarity given
    '''
    
    classes = Pk
    classes.flags.writeable = False
    print("\tClassifying... (Similarity : "+sim + " _ " + param + ")")

    

    doc_results = []
    doc_maxes = []


    lamba = lamba_
    if lamba_ is None:
        lamba = 1

    
    
    Doconf = []
    for doc_words in IFS:

        results = []

        for class_ in Pk:

            

            ifsSim = similarity(sim, doc_words, class_, type=param, omegas=omegas, lamda=lamba)
            results.append(ifsSim)

            doc_results.append(np.argmax(results))  # np.argmin(results))
            doc_maxes.append(np.max(results))

            Doconf.append(np.sum(np.absolute(np.max(results) - results)))
                

            

    Accuracy = accuracy_score(targets, doc_results)
    Precision = precision_score(targets, doc_results, average='micro')
    Recall = recall_score(targets, doc_results, average='micro')
    fpr, tpr, thresholds = roc_curve(targets, doc_results,
                                             pos_label=int(len(np.unique(targets)) / 2))  # ROC curve
    # fpr : Increasing false positive rates
    # tpr : Increasing true positive rates
    # thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
    try:
        Roc = auc(tpr, fpr)  # auc score
    except:
        Roc = 0.01

    Doconf = np.absolute(Doconf - np.mean(Doconf)) / np.std(Doconf)

    Similarity=[sim, param, np.mean(doc_maxes), np.mean(Doconf), Accuracy, Precision, Recall, Roc]
            

    return Similarity
    



