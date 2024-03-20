@author: nikolaos damianos
#######################################################################################################################
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif
#from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
from Measures.Distances import  Distances_
from Measures.Similarities import  Similarities_
from sklearn.model_selection import train_test_split, KFold
#import re
from textblob import Word

from Classify.Helper import classes_str_to_int,clean_str
from Classify.Fuzzy_Classify import dist_param , sim_param  ,cross_validate

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



#########################################################################################################################
##########################################################################################################################
###########################################################################################################################



if __name__ == '__main__':
    #load dataset
    bbc_data = pd.read_csv('data/dataset.csv', encoding="ISO-8859-1")
    bbcsport_data = pd.read_csv('data/dataset2.csv', encoding="ISO-8859-1")
    
    #split data 
    all_data = [bbcsport_data, bbc_data]

    for d_i in range(2):
        
        print(all_data[d_i])
        
        
        data = all_data[d_i]
        x = data['news'].tolist()  # the documents
        y = data['type'].tolist()  # the classes

        # gia ta bbc datasets
        print("processing data")
        for index, value in enumerate(x):
            x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

        count_vect = CountVectorizer(min_df=0)
        # print('count_vect : ' , str(count_vect))

        vect = count_vect.fit_transform(x)

        information_gain = mutual_info_classif(vect, y, discrete_features=True)#Estimated mutual information between each feature and the target.
        half = int(information_gain.shape[0] * 0.5)
        sorted_gain_indexes = np.argsort(-np.array(information_gain))
        top_half_indexes = sorted_gain_indexes[:half]
        top_half_words = count_vect.get_feature_names()[:half]
        count_vect = CountVectorizer(min_df=0, vocabulary=top_half_words)
        vect = count_vect.fit_transform(x)

        X = vect.toarray()
        Y = np.array(classes_str_to_int(y))

        # # get the vocabulary of the training set (all the words)
        train_vocabulary = count_vect.vocabulary_
        # Load test dataset
        # get only the words that were used in the training process (vocabulary=train_vocabulary)

        # "train" with the dataset. returns array that contains for each class the membership/non-membership/hesitation degree
        # of each word.
        Dis = {}
        Sim = {}
        k_fold = 10

        all_dists = Distances_()
        all_sims = Similarities_()
        
        
        

        best_per_fold_dists = []#the best of every distance/param  
        for d in Distances_():
            for _ in dist_param()[d]:
                
                best_per_fold_dists.append([-1 for _ in range(9)])

        best_per_fold_sims = []#the best of every similarity/param  
        for d in Similarities_():
            for _ in sim_param()[d]:
                best_per_fold_sims.append([-1 for _ in range(9)])

        rj1s = np.arange(0.6, 1.01, 0.1)
        rj2s = np.arange(0.6, 1.01, 0.1)
        
        #np.random.shuffle(rj1s)
        #np.random.shuffle(rj2s)
        
        #k_fold = 10
        kf = KFold(n_splits=k_fold, shuffle=True)
        splits = list(kf.split(X))
        train_test_indexes = [splits[cv] for cv in range(k_fold)]

        cross_validation_data = [(X[train_index], Y[train_index], X[test_index], Y[test_index]) for train_index, test_index in train_test_indexes]

        from multiprocessing import Pool
        from itertools import product
        
        for rj1 in rj1s:
            for rj2 in rj2s:
                
                key_name = 'rj1= ' + str(rj1) + ' rj2= ' + str(rj2)
                print(key_name)
                cv_args = product([rj1], [rj2], cross_validation_data)
                with Pool(8) as p:
                    results = p.starmap(cross_validate, cv_args)

                cv_results_dists = np.asarray([r[0] for r in results])
                cv_results_sims = np.asarray([r[1] for r in results])

                mean_dist = np.array(cv_results_dists[:, :, 2:], dtype=np.float)
                #print("mean_dist = " + str(mean_dist))
                
                mean_dist = np.mean(mean_dist, axis=0)
                
                
                
                for i in range(mean_dist.shape[0]):#for every distance i = distance/parameter index 
                    #print(float(best_per_fold_dists[i][5]))
                    best_acc = float(best_per_fold_dists[i][5])
                    #print(best_acc)
                    current_acc = mean_dist[i][2]
                    
                    
                    
                    if best_acc <= current_acc:
                        best_per_fold_dists[i] = np.append([cv_results_dists[0][i][0], cv_results_dists[0][i][1], "rj1={}, rj2={}".format(rj1, rj2)], mean_dist[i])
                        print("best_per_fold_dists : " + str(best_per_fold_dists[i]))

                mean_sim = np.array(cv_results_sims[:, :, 2:], dtype=np.float)
                mean_sim = np.mean(mean_sim, axis=0)
                
                
                for i in range(mean_sim.shape[0]):
                    print(i)
                    best_acc = float(best_per_fold_sims[i][5])
                    
                    
                    current_acc = mean_sim[i][2]
                    print("current_acc : " + str(current_acc))
                    if best_acc < current_acc:
                        best_per_fold_sims[i] = np.append(
                            [cv_results_sims[0][i][0], cv_results_sims[0][i][1], "rj1={}, rj2={}".format(rj1, rj2)],
                            mean_sim[i])

        dt = pd.DataFrame(best_per_fold_dists)
        dt.columns = ["Measure", "Parameters", "Best r pars", "MValue", "Degree of Confidence", "Accuracy", "Precision",
                      "Recall", "ROC"]
        dt.to_csv("dists_{}.csv".format(d_i))

        dt = pd.DataFrame(best_per_fold_sims)
        dt.columns = ["Measure", "Parameters", "Best r pars", "MValue", "Degree of Confidence", "Accuracy", "Precision",
                      "Recall", "ROC"]
        dt.to_csv("sims_{}.csv".format(d_i))
