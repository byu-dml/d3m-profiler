import numpy as np
#import multiprocessing as mp
import os
import pathlib as pl
import pandas as pd
import time
import pickle
import sys
from mpi4py import MPI
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#from sklearn.neural_network import MLPClassifier as MLPClassifier
#from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QuadraticDiscriminantAnalysis

  
def save_results(results,conf):
    #save the results to a csv file
    results.to_csv(model_name+'_final_cross_val.csv',index=False)
    filename = model_name+'_matrix_mean.pkl'
    fileObject = open(filename, 'wb')
    pickle.dump(conf, fileObject)
    fileObject.close()
    
def naive_gen():
    model_name = 'Naive'
    class NaiveModel:
        def fit(self,X_train,y_train):
            self.majority = y_train.value_counts().idxmax()
        def predict(self,X_test):
            y_hat = [self.majority for i in range(len(X_test))]
            return y_hat
    model = NaiveModel()
    return model, model_name
    
def fit_predict_model(X_data, y, train_ind, test_ind, balance=True):
    #now fit using the indeces given by the kfold splitter
    X_train = X_data[train_ind]
    y_train = y.iloc[train_ind]
    #get the labels for the confusion matrix
    if (balance == True):
        #get the k_neighbors balance number
        k_neighbors = y_train.value_counts().min()-1
        assert k_neighbors > 0, 'Not enough data to rebalance. Must be more than 1:.'
        #rebalance
        #print("balancing")
        begin = time.time()
        smote = BorderlineSMOTE(sampling_strategy='not majority',k_neighbors=k_neighbors,random_state=32)
        X_train, y_train = smote.fit_resample(X_train,y_train)
        end = time.time()
        print("Time to rebalance: "+str(np.round(end-begin,3)))       
    #fit on  data
    model.fit(X_train,y_train)
    #predict on the model
    del X_train
    del y_train
    y_hat = list(model.predict(X_data[test_ind]))
    y_test = list(y.iloc[test_ind])
    print("Finished Fold "+str(rank))
    return y_hat, y_test
    
def compile_results(results_final: list):
    y_test = list()
    y_hat = list()
    #compute the results
    for hat, test in results_final:
        y_test += test
        y_hat += hat
    accuracy = accuracy_score(y_test, y_hat)
    f1_macro = f1_score(y_test, y_hat, average='macro')
    f1_micro = f1_score(y_test, y_hat, average='micro')
    f1_weighted = f1_score(y_test, y_hat, average='weighted')
    conf = confusion_matrix(y_test, y_hat) 
    results = pd.DataFrame(data=np.array([[model_name, accuracy, f1_macro, f1_micro, f1_weighted]]), columns=['classifier', 'accuracy_score', 'f1_score_macro', 'f1_score_micro', 'f1_score_weighted'])
    save_results(results, conf) 
    return results
    
def get_from_csv(data_file: str):
    data = pd.read_csv(data_file)
    X_data = data.drop(['colType','datasetName'],axis=1).to_numpy()
    y = data['colType']
    groups = data['datasetName']
    return X_data, y, groups
    
def get_jobs_list(X_data, groups, size: int, cross_type):
    splitter = cross_type
    jobs = list(splitter.split(X_data, groups=groups))
    list_jobs_total = [list() for i in range(COMM.size)]
    for i in range(len(jobs)):
        j = i % COMM.size
        list_jobs_total[j].append(jobs[i])
    return list_jobs_total
    
def get_variables(use_col_name_only, use_metadata, rank):
    if (rank == 0):
        if (use_metadata):
            if (use_col_name_only is True):
                closed_embed = 'closed_embed_lower.csv' 
                X_data, y, groups = get_from_csv(data_file = closed_embed)   
            else:
                closed_embed = 'closed_embed_all.csv'
                X_data, y, groups = get_from_csv(data_file = closed_embed)
        else:
            X_data, y, groups = parse_dataset(get_datasets(DATASET_DIR))     
        #format jobs list to split across processors
        jobs = get_jobs_list(X_data = X_data, groups = groups, size=COMM.size, cross_type=LeaveOneGroupOut())  
    else:
        if (use_metadata):
            if (use_col_name_only is True):
                X_data = np.empty((47831, 768), dtype='d')
            else:
                X_data = np.empty((47831, 768*3), dtype='d')
        else:
            X_data = np.empty((shape_SIMON), dtype='d')
        jobs = None
        y = None
    
    return X_data, jobs, y
         
def evaluate_model(balance: bool, use_col_name_only: bool, use_metadata: bool, rank=None):   
    #get the variables for cross validation
    X_data, jobs, y = get_variables(use_col_name_only = use_col_name_only, use_metadata = use_metadata, rank = rank)
    #get the values from the root processor
    y = COMM.bcast(y,root=0)
    jobs = COMM.scatter(jobs, root=0)
    COMM.Bcast([X_data, MPI.FLOAT], root=0)
    #run cross-validation on all the different processors
    results_init = []
    for job in jobs:
        train_ind, test_ind = job
        results_init.append(fit_predict_model(X_data,y,train_ind, test_ind, balance=balance))
    #gather results from processors
    results_init = MPI.COMM_WORLD.gather(results_init, root = 0)
    del jobs

    #compile and save the results
    if (rank == 0):
        del X_data
        del y
        print("Finished cross validation!")
        results = compile_results(results_final = [_i for temp in results_init for _i in temp])
 
if __name__ == "__main__":   
    random_state = 32
    #model_name = 'RF_PCA_lower_border'
    #pca = PCA(n_components='mle',random_state=random_state)
    #rf = RandomForestClassifier(random_state=random_state)
    #model = Pipeline(steps=[('pca',pca),('rf',rf)])
    model, model_name = naive_gen()  
    COMM = MPI.COMM_WORLD
    rank = COMM.rank
    results = evaluate_model(balance=False, use_col_name_only=True, use_metadata=True, rank=rank)
