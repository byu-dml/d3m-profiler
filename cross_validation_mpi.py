import numpy as np
#import multiprocessing as mp
import os
import pathlib as pl
import pandas as pd
#from d3m_profiler.build_table import get_datasets
import json
import time
import pickle
import sys
from mpi4py import MPI
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
#from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
#from sklearn.neural_network import MLPClassifier as MLPClassifier
#from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB as GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QuadraticDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.decomposition import PCA
#from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier as MLPClassifier

DATASET_DIR = '/users/data/d3m/datasets/training_datasets'
METADATA_PATH = '~/data/closed_d3m_unembed_data.csv'
embed_col_path = ''
embed_all_path = ''
MAX_LEN = 20
MAX_CELLS = 100

def parse_datasets(datasets):
    raw_data, header, groups = [], [], []
    for dataset_id, dataset_doc_path in datasets.items():
        # open the dataset doc to get the column headers
        with open(dataset_doc_path, 'r') as dataset_doc:
            meta_dataset = json.load(dataset_doc)
            for resource in meta_dataset['dataResources']:
                if 'columns' not in resource:
                    continue
                # then open the actual dataset table to get column values
                if resource['resPath'][-4:] == '.csv':
                    try:
                        dataset = pd.read_csv(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath']))
                    except:
                        continue
                else:
                    values = 0
                    tables = []
                    for entry in os.scandir(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath'])):
                        tables.append(pd.read_csv(entry.path))
                        values += len(tables[-1])
                        if values >= MAX_CELLS:
                            break
                    dataset = pd.concat(tables, ignore_index=True)

                for column in resource['columns']:
                    values = list(dataset[column['colName']].values)
                    if len(values) > MAX_CELLS:
                        values = [str(v)[:MAX_LEN] for v in values[:MAX_CELLS]]
                    else:
                        values = [str(v)[:MAX_LEN] for v in values] + ['' for i in range(MAX_CELLS - len(values))]

                    raw_data.append(values)
                    header.append((column['colType'],))
                    groups.append(meta_dataset['about']['datasetName'])
                    
    return np.asarray(raw_data), np.asarray(header), np.asarray(groups)
  
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
    return model

def evaluate_model(balance: bool, col_name: bool, use_metadata: bool, rank=None):

    def run_fold(train_ind, test_ind, balance=True):
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
            smote = SMOTE(k_neighbors=k_neighbors)
            X_train, y_train = smote.fit_resample(X_train,y_train)       
        #fit on  datai
        #print("fitting model")
        model.fit(X_train,y_train)
        #predict on the model
        del X_train
        del y_train
        y_hat = list(model.predict(X_data[test_ind]))
        y_test = list(y.iloc[test_ind])
        print("Finished Fold "+str(rank))
        return y_hat, y_test
  
    if (rank == 0):
        print("Beginning cross validation")
        if (use_metadata):
            if (col_name is True):
                type_column = 'colType'
                closed_embed = 'embedded_d3m_closed.csv'    
                print("loading file")
                data = pd.read_csv(closed_embed)
                print("Done loading!")
                #do shuffled cross validation, but that can also be replicated
                X_data = data.drop(['colType','datasetName'],axis=1).to_numpy()
                y = data['colType']
                groups = data['datasetName']
            else:
                type_column = 'colType'
                closed_embed = 'embedded_d3m_closed_all.csv'
                print("loading file")
                data = pd.read_csv(closed_embed)
                print("Done loading!")
                X_data = data.drop(['colType','datasestName'],axis=1).to_numpy()
                y = data['colType']
                groups = data['datasetName']
        else:
            X_data, y, groups = parse_dataset(get_datasets(DATASET_DIR))
            
        k_splitter = LeaveOneGroupOut()
        #split_num = k_splitter.get_n_splits(X_data, groups=groups)
        jobs = list(k_splitter.split(X_data, groups=groups))
        #gets the jobs and splits them into even-sized-lists to be spread across the different cpu's
        list_jobs_total = [list() for i in range(COMM.size)]
        for i in range(len(jobs)):
            j = i % COMM.size
            list_jobs_total[j].append(jobs[i])
        jobs = list_jobs_total
    else:
        #initalizes variables to pass to other processors, size of X_data must be intialized correctly
        if (use_metadata):
            if (col_name is True):
                X_data = np.empty((47831, 768),dtype='d')
            else:
                print("bad")
                X_data = np.empty((47831, 768*3), dtype='d')
        else:
                X_data = np.empty((shape_data), dtype='d')  
        y = None
        jobs = None
           
    #get the values from the root processor
    y = COMM.bcast(y,root=0)
    jobs = COMM.scatter(jobs, root=0)
    COMM.Bcast([X_data, MPI.FLOAT], root=0)

    #run cross-validation on all the different processors
    results_init = []
    for job in jobs:
        train_ind, test_ind = job
        results_init.append(run_fold(train_ind, test_ind, balance=balance))

    #gather results together
    results_init = MPI.COMM_WORLD.gather(results_init, root = 0)
    del jobs

    #compile and save the results
    if (rank == 0):
        del X_data
        del y
        print("Finished cross validation!")
        #flatten the total results
        results_final = [_i for temp in results_init for _i in temp]
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
        results = pd.DataFrame(data=[model_name, accuracy, f1_macro, f1_micro, f1_weighted], columns=['classifier', 'accuracy_score', 'f1_score_macro', 'f1_score_micro', 'f1_score_weighted'])
        print(results)
        save_results(results, conf) 
        return results
    
if __name__ == "__main__":
    random_state = 32
    model_name = 'GNB'
    model = GaussianNB() 
    COMM = MPI.COMM_WORLD
    rank = COMM.rank
    evaluate_model(balance=True, col_name=True, use_metadata=True, rank=rank)
    

