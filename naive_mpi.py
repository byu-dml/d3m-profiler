import numpy as np
#import multiprocessing as mp
import os
import pathlib as pl
import pandas as pd
import pickle
import sys
from mpi4py import MPI
from imblearn.over_sampling import SMOTE
#from d3m_profiler import rebalance
#from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB as GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QuadraticDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.decomposition import PCA
#from sklearn.neural_network import MLPClassifier as MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score


#define the model
model_name = 'Naive'
class NaiveModel:
    def fit(self,X_train,y_train):
        self.majority = y_train.value_counts().idxmax()
    def predict(self,X_test):
        y_hat = [self.majority for i in range(len(X_test))]
        return y_hat
model = NaiveModel()

def run_fold(train_ind, test_ind, balance=True):
    #now fit using the indeces given by the kfold splitter
    X_train_embed = X_embed[train_ind]
    y_train = y.iloc[train_ind]
    labels = y_train.unique()
    #get the labels for the confusion matrix
    if (balance == True):
        #get the k_neighbors balance number
        k_neighbors = y_train.value_counts().min()-1
        assert k_neighbors > 0, 'Not enough data to rebalance. Must be more than 1:.'
        #rebalance
        smote = SMOTE(k_neighbors=k_neighbors)
        X_train_embed, y_train = smote.fit_resample(X_train_embed,y_train)
        
    #fit on  data
    model.fit(X_train_embed,y_train)
    #predict on the model
    y_hat = model.predict(X_embed[test_ind])
    y_test = y.iloc[test_ind]
    #get the scores of results
    f1_macro = f1_score(y_test, y_hat, average='macro')
    f1_micro = f1_score(y_test, y_hat, average='micro')
    f1_weighted = f1_score(y_test, y_hat, average='weighted')
    accuracy = accuracy_score(y_test, y_hat)
    conf = confusion_matrix(y_test, y_hat, labels=labels)
    print("Finished Fold!")
    return f1_macro, f1_micro, f1_weighted, accuracy, conf
  
COMM = MPI.COMM_WORLD
if (COMM.rank == 0):
    print("Beginning cross validation")
    type_column = 'colType'
    closed_embed = 'embedded_d3m_closed.csv'
    
    print("loading file")
    embed_df = pd.read_csv(closed_embed)
    print("Done loading!")
    
    #do shuffled cross validation, but that can also be replicated
    X_embed = embed_df.drop(['colType','datasetName'],axis=1).to_numpy()
    y = embed_df['colType']
    dataset_names = embed_df['datasetName']
    k_splitter = LeaveOneGroupOut()
    split_num = k_splitter.get_n_splits(embed_df, groups=dataset_names)
    jobs = list(k_splitter.split(embed_df, groups=dataset_names))
    
    #gets the jobs and splits them into even-sized-lists to be spread across the different cpu's
    list_jobs_total = [list() for i in range(COMM.size)]
    for i in range(len(jobs)):
        j = i % COMM.size
        list_jobs_total[j].append(jobs[i])
    jobs = list_jobs_total
else:
    #initalizes variables to pass to other processors, size of X_embed is intialized correctly
    X_embed = np.empty((47831, 768),dtype='d')
    y = None
    jobs = None

#get the values from the root processor
y = COMM.bcast(y,root=0)
jobs = COMM.scatter(jobs, root=0)
COMM.Bcast([X_embed, MPI.FLOAT], root=0)

#run cross-validation on all the different processors
results_init = []
for job in jobs:
    train_ind, test_ind = job
    results_init.append(run_fold(train_ind, test_ind, balance=False))

#gather results together
results_init = MPI.COMM_WORLD.gather(results_init, root = 0)
jobs = list()

#compile and save the results
if (COMM.rank == 0):
    jobs = list()
    results = pd.DataFrame(columns=['classifier', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])
    print()
    print("Finished cross validation!")
    print()
    print("Start compiling answer...")

    results_final = [_i for temp in results_init for _i in temp]
    
    f1s_macro = list()
    f1s_micro = list()
    f1s_weighted = list()
    accuracys = list()
    confusions = list()
    for f1_macro, f1_micro, f1_weighted, accuracy, conf in results_final:
        f1s_macro.append(f1_macro)
        f1s_micro.append(f1_micro)
        f1s_weighted.append(f1_weighted)
        accuracys.append(accuracy)         
        confusions.append(conf)

    mean_f1_macro = np.mean(f1s_macro)    
    mean_f1_micro = np.mean(f1s_micro)
    mean_f1_weighted = np.mean(f1s_weighted)
    mean_accuracy = np.mean(accuracys) 
    print({'classifier': model_name, 'accuracy_score': mean_accuracy, 'f1_score_micro': mean_f1_micro,      'f1_score_macro': mean_f1_macro, 'f1_score_weighted': mean_f1_weighted})   
        
    results = results.append({'classifier': model_name, 'accuracy_score': mean_accuracy, 'f1_score_micro': mean_f1_micro, 'f1_score_macro': mean_f1_macro, 'f1_score_weighted': mean_f1_weighted}, ignore_index=True) 

    results.to_csv(model_name+'_final_cross_val.csv',index=False)
    conf_mean = np.sum(confusions) / len(confusions)
    filename = model_name+'_matrix_mean.pkl'
    fileObject = open(filename, 'wb')
    pickle.dump(conf_mean, fileObject)
    fileObject.close()
