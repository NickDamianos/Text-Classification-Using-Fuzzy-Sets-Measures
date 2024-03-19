from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import roc_curve,auc
import pandas as pd
import numpy as np
from Measures.Distances import distance , Distances
from Measures.Similarities import similarity, Similarities
import xlsxwriter
from sklearn.model_selection import train_test_split
import re
from textblob import Word

#ka8arizei to dataset(aferi apo ta datasets ta parakatw)
def clean_str(string):    
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


#helper function
def classes_str_to_int(targets):
    v = {}

    uni = np.unique(targets)

    for i in range(len(uni)):
        v[uni[i]]=i
    
    ret_targets = []
    for i in range(len(targets)):
        ret_targets.append(v[targets[i]])
    
    
    
    return ret_targets


def dist_param():
    distances = Distances()
    params = {}
    for dist in distances:
        if not dist == 'vlachSergDistance':
            if dist == 'wangXinDistance':
                params[dist] = [1 , 2]
            else:
                params[dist] = ['H' , 'E' , 'nH' , 'nE']
        else:
            params[dist] = '-'

    return params


def sim_param():
    similarities = Similarities()
    params = {}

    for sim in similarities:
        if sim in [similarities[1], similarities[4] , similarities[5] , similarities[8] , similarities[10] , similarities[12] , similarities[16]]:
            if sim == similarities[1]:
                params[sim] = ['e','s','h']
            elif sim == similarities[4]:
                params[sim] = [1 , 2]
            elif sim in [similarities[5],similarities[8],similarities[12]]:
                params[sim] = ['l','e','c']
            elif sim == similarities[10]:
                params[sim] = ['w1','w2','pk1','pk2','pk3','new1','new2']
            else:
                params[sim] = [i for i in range(1, 21)]
        else:
            params[sim] = '-'

    return params

    
def calculate_membership(data, rj1, rj2, Xj=None, sj=None):
    print(data.shape)
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
    print ("Preprocessing...")
    
    docs, words = np.array(data).shape
    print(docs, words)
    print ("\tCalculating (non)membership values...")
    m, v, p, Xj, sj = calculate_membership(data, rj1, rj2)

    print ("\tCalculating class patterns...")
    numberOfClasses = len(np.unique(targets))

    # Pk[0] - > first class
    # Pk[2][0] -> values (membership/etc) of the first word for the third class
    # Pk[3][0][1] -> non-membership value of the first word for the forth class
    Pk = np.zeros((numberOfClasses, words, 3), dtype=np.float64) # 3 for m v p

    classes = np.unique(targets)

    docs = np.asarray(range(docs))

    for c in range(len(classes)):
        same_class_docs = docs[targets == classes[c]]
        
        Pk[c, :, 0] = np.average(m[same_class_docs, :], axis=0)
        
        Pk[c, :, 1] = np.average(v[same_class_docs, :], axis=0)
        
        Pk[c, :, 2] = np.average(p[same_class_docs, :], axis=0)
        
    return Pk, Xj, sj


def fuzzy_test(test_data, Xj, sj, rj1, rj2):
    print ("Testing...")

    print ("\tCalculating (non)membership values...")
    docs, words = test_data.shape

    A = np.zeros((docs, words, 3), dtype=np.float64)
    # calculate the membership values of each word for each document
    for doc in range(docs):
        A[doc, :, 0], A[doc, :, 1], A[doc, :, 2], _, _ = calculate_membership(test_data[doc, :], rj1, rj2, Xj, sj)


    return A


def f_similarity(A, B):
    h = A.shape[0]

    s = np.sum((A[:, 0] * B[:, 0] + A[:, 1] * B[: ,1]) /
               (np.sqrt(np.square(A[:, 0]) + np.square(A[:, 1])) * np.sqrt(np.square(B[:, 0]) + np.square(B[:, 1]))))

    return s/h


def fuzzy_classify_all_Dist(Pk, IFS,targets=None):
    classes = Pk
    classes.flags.writeable = False

    print("Classifying... (Distancies)")
    distances = []

    doc_results = []
    doc_mins = []

    dis = Distances()
    params = dist_param()

    
    i = 0
    for d in dis:
        #print('Dist ' + str(i) + " / " + str(cnt))
        i += 1
        par = params[d]
                
        for p in par:
            Doconf = []
            doc_cnt = 0
            for doc_words in IFS: # for each document
                #print('Dist ' + str(i) + " / " + str(cnt), '  ', p, ' , ', 'Doc ' + str(doc_cnt) + '/' + str(len(IFS)-1))
                results = []
                

                for class_ in Pk: # for each class
                    # calculate the distance between the class and the current document doc_words
                    if p is None:
                        p = 1

                    ifsDist = distance(d, doc_words, class_ , type=p)
                    results.append(ifsDist) #append the distance value
                    
                # get the class that has the mimimum distance with the current document
                doc_results.append(np.argmin(results))#np.argmin(results))
                doc_mins.append(np.min(results))

                Doconf.append(np.sum(np.absolute(np.min(results)-results)))
                doc_cnt+=1

                # Doconf = []

            print(d + ' ' + str(p) + ' Acuracy : ', accuracy_score(targets,doc_results))
            #f = error_calc1(doc_results, targets)##classes[mins])#, List=True)

            Accuracy = accuracy_score(targets, doc_results)
            Precision = precision_score(targets, doc_results,average='macro')
            Recall =  recall_score(targets, doc_results,average='macro')
            fpr, tpr, thresholds = roc_curve(targets, doc_results, pos_label=int(len(np.unique(targets))/2))#ROC curve
            
            #fpr : Increasing false positive rates
            #tpr : Increasing true positive rates
            #thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
            Roc = auc(fpr, tpr) # auc score
                   
            Doconf = np.absolute(Doconf - np.mean(Doconf))/np.std(Doconf) 
            distances.append([d, p, np.mean(doc_mins), np.mean(Doconf), Accuracy, Precision, Recall, Roc])#roc_auc_score(targets,doc_results)])
            doc_results = []
            doc_mins=[]

        

    return distances


def fuzzy_classify_all_Similarities(Pk, IFS , lamba_ = None,targets=None):
    
    classes = Pk
    classes.flags.writeable = False
    print("Classifying... (Similarities)")
    
    similarities = []

    doc_results = []
    doc_mins = []

    params = sim_param()
    sims = Similarities()

    omegas = [0.5 , 0.3 , 0.2]

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
                    
                    ifsSim = similarity(s,doc_words, class_ , type=p , omegas=omegas, lamda=lamba)
                    results.append(ifsSim)
                    
                
                doc_results.append(np.argmax(results))#np.argmin(results))
                doc_mins.append(np.max(results))

                Doconf.append(np.sum(np.absolute(np.max(results)-results)))
                doc_cnt+=1
            
            print(s + ' '+ str(p) + ' Acuracy :', accuracy_score(targets,doc_results))
            

            Accuracy = accuracy_score(targets, doc_results)
            Precision = precision_score(targets, doc_results,average='micro')
            Recall =  recall_score(targets, doc_results,average='micro')
            fpr, tpr, thresholds = roc_curve(targets, doc_results, pos_label=int(len(np.unique(targets))/2))#ROC curve
            #fpr : Increasing false positive rates
            #tpr : Increasing true positive rates
            #thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr
            try:
                Roc = auc(tpr, fpr) # auc score
            except :
                Roc = 0.01
            Doconf = np.absolute(Doconf - np.mean(Doconf))/np.std(Doconf) 
            
            similarities.append([s, p, np.mean(doc_mins), np.mean(Doconf), Accuracy, Precision, Recall, Roc])
            doc_results = []
            doc_mins = []
            
    return similarities
                

def fuzzy_classify(Pk, IFS):
    print ("Classifying...")
    doc_results = []
    doc_mins = []

    for doc_words in IFS: # for each document
        results = []

        for class_ in Pk: # for each class
            # calculate the similarity between the class and the current document doc_words
            results.append(f_similarity(doc_words, class_))
            print(results[:-1])

        # get the class that has the maximum similarity with the current document
        doc_results.append(np.argmax(results))
        doc_mins.append(np.max(results))
    return doc_results, doc_mins


def excel_(Dataset, xx):
    
    #apo8ikevei se excel
    address = './Excel/'+ Dataset + '.xlsx'
    #new_addr_excel = t + '/newresults' + str(i) + '.xlsx'
    excel = xlsxwriter.Workbook('./Excel/'+ Dataset + '.xlsx')
    excel_work = excel.add_worksheet()
    xx = np.array(xx)
    headers = ['Function Name','Type', 'MValue' ,'Degree of Confidence' , 'Accuracy','Specificity','Sensitivity' ,'ROC AUC']
    #[dis[i] , par[j] , np.argmin(results) , np.min(results) , Doconf , f[0] , f[1] , f[2]]
    excel_work.write('A1', headers[0]+"  ")
    excel_work.write('B1', headers[1]+"  ")
    excel_work.write('C1', headers[2]+"  ")
    excel_work.write('D1', headers[3]+"  ")
    excel_work.write('E1', headers[4]+"  ")
    excel_work.write('F1', headers[5]+"  ")
    excel_work.write('G1', headers[6]+"  ")
    excel_work.write('H1', headers[7]+"  ")
    
    excel_work.write('A2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'A' + str(yy)
        excel_work.write(col, xx[yy - 3][0])

    excel_work.write('B2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'B' + str(yy)
        excel_work.write(col, xx[yy - 3][1])

    excel_work.write('C2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'C' + str(yy)
        excel_work.write(col, xx[yy - 3][2])

    excel_work.write('D2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'D' + str(yy)
        excel_work.write(col, str(xx[yy - 3][3]))

    excel_work.write('E2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'E' + str(yy)
        excel_work.write(col, str(xx[yy - 3][4]))

    excel_work.write('F2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'F' + str(yy)
        excel_work.write(col, str(xx[yy - 3][5]))

    excel_work.write('G2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'G' + str(yy)
        excel_work.write(col, str(xx[yy - 3][6]))
        
    excel_work.write('H2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'H' + str(yy)
        excel_work.write(col, xx[yy - 3][7])
    
    excel.close()
    
    df = pd.read_excel(address)
    print(df)
    data_xls = pd.read_excel(address, 'Sheet1', index_col=None)
    data_xls.to_csv('CSV/'+Dataset + '.csv', encoding='utf-8')
    
def excel_3 (d,typeI):
    workbook = xlsxwriter.Workbook('best_rjs_'+typeI+'.xlsx')
    worksheet = workbook.add_worksheet()

    
    row = 0
    col = 0

    for key in sorted(d.keys()):
        row += 1
        worksheet.write(row, col, key)
        oo = 1
        for item in d[key]:
            worksheet.write(row, col+oo , item)
            
            oo+=1
        row += 1
    workbook.close()
    data_xls = pd.read_excel('best_rjs_'+typeI+'.xlsx', 'Sheet1', index_col=None)
    data_xls.to_csv('best_rjs_'+typeI+'.csv', encoding='utf-8')

    
    
def excel_2(Dataset, xx,path):
    
    #apo8ikevei se excel
    address = path +'excel/'+ Dataset + '.xlsx'
    #new_addr_excel = t + '/newresults' + str(i) + '.xlsx'
    excel = xlsxwriter.Workbook(address)
    excel_work = excel.add_worksheet()
    xx = np.array(xx)
    headers = ['Rjs','Function Name','Type', 'MValue' ,'Degree of Confidence' , 'Accuracy','Specificity','Sensitivity' ,'ROC AUC']
    #[dis[i] , par[j] , np.argmin(results) , np.min(results) , Doconf , f[0] , f[1] , f[2]]
    excel_work.write('A1', headers[0]+"  ")
    excel_work.write('B1', headers[1]+"  ")
    excel_work.write('C1', headers[2]+"  ")
    excel_work.write('D1', headers[3]+"  ")
    excel_work.write('E1', headers[4]+"  ")
    excel_work.write('F1', headers[5]+"  ")
    excel_work.write('G1', headers[6]+"  ")
    excel_work.write('H1', headers[7]+"  ")
    excel_work.write('I1', headers[8]+"  ")
    
    excel_work.write('A2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'A' + str(yy)
        excel_work.write(col, str(xx[yy - 3][0]))

    excel_work.write('B2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'B' + str(yy)
        excel_work.write(col, xx[yy - 3][1])

    excel_work.write('C2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'C' + str(yy)
        excel_work.write(col, xx[yy - 3][2])

    excel_work.write('D2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'D' + str(yy)
        excel_work.write(col, str(xx[yy - 3][3]))

    excel_work.write('E2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'E' + str(yy)
        excel_work.write(col, str(xx[yy - 3][4]))

    excel_work.write('F2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'F' + str(yy)
        excel_work.write(col, str(xx[yy - 3][5]))

    excel_work.write('G2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'G' + str(yy)
        excel_work.write(col, str(xx[yy - 3][6]))
        
    excel_work.write('H2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'H' + str(yy)
        excel_work.write(col, xx[yy - 3][7])
    
    excel_work.write('I2', '------------')

    for yy in range(3, len(xx) + 3):
        col = 'H' + str(yy)
        excel_work.write(col, xx[yy - 3][8])
    
    excel.close()
    
    #df = pd.read_excel(address)
    
    data_xls = pd.read_excel(address, 'Sheet1', index_col=None)
    data_xls.to_csv(path+'csv/'+Dataset + '.csv', encoding='utf-8')
    

    
#########################################################################################################################
##########################################################################################################################
###########################################################################################################################
#bbc datasets load
'''    
bbc_data = pd.read_csv('C:/Users/nikolaos damianos/Desktop/hh1/data/dataset.csv',encoding = "ISO-8859-1")
bbcsport_data= pd.read_csv('C:/Users/nikolaos damianos/Desktop/hh1/data/dataset2.csv',encoding = "ISO-8859-1")

x = bbcsport_data['news'].tolist()#ta keimena 
y = bbcsport_data['type'].tolist()#oi classes

rj1, rj2 = 0.9, 0.9 

#gia ta bbc datasets

import re
from textblob import Word

#ka8arizei to dataset(aferi apo ta datasets ta parakatw)
def clean_str(string):    
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


for index,value in enumerate(x):
    print( "processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])



count_vect = CountVectorizer(min_df=0)
print('count_vect : ' , str(count_vect))



vect = count_vect.fit_transform(x)#TfidfVectorizer(stop_words='english',min_df=2)
X = vect.toarray() #vect.fit_transform(x)
Y = np.array(classes_str_to_int(y))

print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# # get the vocabulary of the training set (all the words)
train_vocabulary = count_vect.vocabulary_
# # "train" with the dataset. returns array that contains for each class the membership/non-membership/hesitation degree
# # of each word.
Dis = {}
Sim = {}

#class_patterns, Xj, sj =fuzzy_train(X, Y , rj1, rj2)# fuzzy_train(counts_array, twenty_train.target, rj1, rj2)
    

range1=np.arange(0.6, 1.01, 0.1)
for rj1 in range1:
    # fuzzy_train(counts_array, twenty_train.target, rj1, rj2)
    for rj2 in range1:
        class_patterns, Xj, sj =fuzzy_train(X_train, y_train , rj1, rj2)
        print(rj1)
        print(rj2)
        key_name = 'rj1= ' + str(rj1) + ' rj2= ' + str(rj2)
        print(key_name)
        # # Load test dataset
        # # get only the words that were used in the training process (vocabulary=train_vocabulary)
        count_vect_test = CountVectorizer(min_df=0, vocabulary=train_vocabulary)
        #counts_test = count_vect_test.fit_transform(X_test)#count_vect_test.fit_transform(twenty_test)
        #counts_array_test = np.asarray(counts_test.toarray())   # row= document, column= word, element= number of occurrences
        X_test = np.asarray(X_test)
        
        # # calculate membership/etc values for each word in each document
        IFS = fuzzy_test(X_test, Xj, sj, rj1, rj2)
        
        
        print('Documents = ', str(len(IFS)))
        # # classify the new documents
        
        Dis[key_name]=fuzzy_classify_all_Dist(class_patterns,IFS,y_test)
        Sim[key_name]=fuzzy_classify_all_Similarities(class_patterns, IFS,targets = y_test)



path = './rj1_2_csvs/' 
for key in Dis.keys():
    excel_2(key,Dis[key],path+'Distancies/')
    excel_2(key,Sim[key],path+'similarities/')

distancies_best_accuracies = {}



for i in range(len(Dis['rj1= 0.7 rj2= 0.7'])): 
    dists = {}
    for key in Dis.keys():
        dists[key] = Dis[key][i][4]
    
    maximum = max(dists,key=dists.get)
    
    distancies_best_accuracies[(Dis['rj1= 0.7 rj2= 0.7'][i][0]+'_'+str(Dis['rj1= 0.7 rj2= 0.7'][i][1]))] = [maximum , dists[maximum]]

    print(maximum , dists[maximum])
    
similarities_best_accuracies = {}

for i in range(len(Sim['rj1= 0.7 rj2= 0.7'])): 
    sims = {}
    for key in Sim.keys():
        sims[key] = Sim[key][i][4]
    
    maximum = max(sims,key=sims.get)
    
    similarities_best_accuracies[(Sim['rj1= 0.7 rj2= 0.7'][i][0]+'_'+str(Sim['rj1= 0.7 rj2= 0.7'][i][1]))] = [maximum , sims[maximum]]

    print(maximum , sims[maximum])


excel_3(distancies_best_accuracies,'Dists')
excel_3(similarities_best_accuracies,'Sims')
'''
'''
excel_('BBCSportDists',Dis)
excel_('BBCSportSims',Sim)
'''

bbc_data = pd.read_csv('data/dataset.csv',encoding = "ISO-8859-1")
bbcsport_data= pd.read_csv('data/dataset2.csv',encoding = "ISO-8859-1")

x = bbcsport_data['news'].tolist()#ta keimena 
y = bbcsport_data['type'].tolist()#oi classes

#gia ta bbc datasets
for index,value in enumerate(x):
    print( "processing data:",index)
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])



count_vect = CountVectorizer(min_df=0)
# print('count_vect : ' , str(count_vect))

vect = count_vect.fit_transform(x)#TfidfVectorizer(stop_words='english',min_df=2)
X = vect.toarray() #vect.fit_transform(x)
Y = np.array(classes_str_to_int(y))



# # get the vocabulary of the training set (all the words)
train_vocabulary = count_vect.vocabulary_
# # "train" with the dataset. returns array that contains for each class the membership/non-membership/hesitation degree
# # of each word.

k_fold = 10

Dis = {}
Sim = {}

for rj1 in np.arange(0.6, 1.01, 0.1):
    for rj2 in np.arange(0.6, 1.01, 0.1):
        all_dists = []
        all_sims = []
        if k_fold != 1:
            indexes = np.arange(X.shape[0], dtype=np.int)
            folds = np.array(np.array_split(indexes, k_fold))
        for cv in range(k_fold):
            print ("CV", cv)
            if k_fold == 1:
                X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
            else:
                test = folds[cv]
                train_indexes = [i for i in range(k_fold) if i != cv]
                train = folds[train_indexes]
                train = [t for subarray in train for t in subarray]
                print ("Training {} chunks, and testing {} chunks".format(len(train), len(test)))
                X_test = X[test]
                y_test = Y[test]
                X_train = X[train]
                y_train = Y[train]

            

            class_patterns, Xj, sj = fuzzy_train(X_train, y_train, rj1, rj2)

            # Load test dataset
            # get only the words that were used in the training process (vocabulary=train_vocabulary)
            count_vect_test = CountVectorizer(min_df=0, vocabulary=train_vocabulary)

            X_test = np.asarray(X_test)

            # # calculate membership/etc values for each word in each document
            IFS = fuzzy_test(X_test, Xj, sj, rj1, rj2)

            print('Documents = ', str(len(IFS)))
            # # classify the new documents
            all_dists.append(fuzzy_classify_all_Dist(class_patterns,IFS,y_test))
            all_sims.append(fuzzy_classify_all_Similarities(class_patterns, IFS,targets = y_test))
        
        key_name = 'rj1= ' + str(rj1) + ' rj2= ' + str(rj2)
        print(key_name)    
        Dis[key_name]= all_dists
        Sim[key_name]= all_sims
            
        
Distan = {}
Similar = {}
#headers = ['rjs', Function Name','Type', 'MValue' ,'Degree of Confidence' , 'Accuracy','Specificity','Sensitivity' ,'ROC AUC']
#                   [s,              p, np.mean(doc_mins), np.mean(Doconf),   Accuracy,   Precision,     Recall,        Roc]

test = 0
for key in sorted(Dis.keys()):
     accuracies = np.zeros(len(Dis[key][0]))
     MValue = np.zeros(len(Dis[key][0]))
     DoC = np.zeros(len(Dis[key][0]))
     Precis = np.zeros(len(Dis[key][0]))
     recall = np.zeros(len(Dis[key][0]))
     Roc = np.zeros(len(Dis[key][0]))
     names = []
     types = []
     for fold in range(len(Dis[key])):

        for dist in range(len(Dis[key][0])):
            accuracies[dist] += Dis[key][fold][dist][4]
            MValue[dist] += Dis[key][fold][dist][2]
            DoC[dist] += Dis[key][fold][dist][3]
            Precis[dist] += Dis[key][fold][dist][5]
            recall[dist] += Dis[key][fold][dist][6]
            Roc[dist] += Dis[key][fold][dist][7]
            names.append(Dis[key][fold][dist][0])
            types.append(Dis[key][fold][dist][1])
        
            
     
     
     n = len(Dis[key][0])
     print('N = ',n,' len = ' , len(DoC))
     di = []
     for i in range(len(Dis[key][0])):
         di.append([key , names[i] , types[i] , MValue[i]/n,DoC[i]/n,accuracies[i]/n,Precis[i]/n,recall[i]/n,Roc[i]/n])
     
     Distan[key] = di  
        
        
for key in sorted(Sim.keys()):
     accuracies = np.zeros(len(Sim[key][0]))
     MValue = np.zeros(len(Sim[key][0]))
     DoC = np.zeros(len(Sim[key][0]))
     Precis = np.zeros(len(Sim[key][0]))
     recall = np.zeros(len(Sim[key][0]))
     Roc = np.zeros(len(Sim[key][0]))
     names = []
     types = []
     for fold in range(len(Sim[key])):

        for dist in range(len(Sim[key][0])):
            accuracies[dist] += Sim[key][fold][dist][4]
            MValue[dist] += Sim[key][fold][dist][2]
            DoC[dist] += Sim[key][fold][dist][3]
            Precis[dist] += Sim[key][fold][dist][5]
            recall[dist] += Sim[key][fold][dist][6]
            Roc[dist] += Sim[key][fold][dist][7]
            names.append(Sim[key][fold][dist][0])
            types.append(Sim[key][fold][dist][1])
        
            
     
     n = len(Sim[key][0])
     print('N = ',n,' len = ' , len(DoC))
     accuracies = accuracies / n
     MValue = MValue/n
     DoC = DoC/n
     Precis = Precis/n
     recall = recall/n
     Roc = Roc/n
     
     
     di = []
     for i in range(n):
         di.append([key , names[i] , types[i] , MValue[i],DoC[i],accuracies[i],Precis[i],recall[i],Roc[i]])
     
     Similar[key] = di        
        
     
        
        
path = './rj1_2_csvs/' 
for key in Distan.keys():
    excel_2(key,Distan[key],path+'Distancies/')
    excel_2(key,Similar[key],path+'similarities/')

distancies_best_accuracies = {}



for i in range(len(Distan['rj1= 0.7 rj2= 0.7'])): 
    dists = {}
    for key in Distan.keys():
        dists[key] = Distan[key][i][5]#accuracy
    
    maximum = max(dists,key=dists.get)
    print(dists[key])
    distancies_best_accuracies[(Distan[maximum][i][1][0]+'_'+str(Distan[maximum][i][2]))] = [maximum , dists[maximum]]

    #print(maximum , dists[maximum])

jl=0
for key in distancies_best_accuracies.keys():
        ff = Distan[distancies_best_accuracies[key][0]]

        excel_2(distancies_best_accuracies[key][0] , ff , path+'bests/')
    
similarities_best_accuracies = {}

for i in range(len(Sim['rj1= 0.7 rj2= 0.7'])): 
    sims = {}
    for key in Similar.keys():
        sims[key] = Similar[key][i][5]
    
    maximum = max(sims,key=sims.get)
    
    similarities_best_accuracies[Similar['rj1= 0.7 rj2= 0.7'][i][0]] = [maximum , sims[maximum]]

    print(maximum , sims[maximum])
jl=0
for key in similarities_best_accuracies.keys():
        ff = Similar[similarities_best_accuracies[key][0]]
        excel_2(similarities_best_accuracies[key][0] , ff , path+'bests/')




