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
from sklearn.neural_network import MLPClassifier as MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score


model = MLPClassifier
model_name = model.__name__
model = model()

def run_fold(train_ind, test_ind):
    #now fit on every fold 
    X_train_embed = X_embed[[train_ind]]
    y_train = y.iloc[train_ind]
    #print("Balancing training data "+str(j))
    k_neighbors = y_train.value_counts().min()-1
    assert k_neighbors > 0, 'Not enough data to rebalance. Must be more than 1:.'
    smote = SMOTE(k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_embed,y_train)
    X_train_embed = list()
    y_train = list()
    model.fit(X_train_bal,y_train_bal)
    y_hat = model.predict(X_embed[[test_ind]])
    y_test = y.iloc[test_ind]
    f1_macro = f1_score(y_test, y_hat, average='macro')
    f1_micro = f1_score(y_test, y_hat, average='micro')
    f1_weighted = f1_score(y_test, y_hat, average='weighted')
    accuracy = accuracy_score(y_test, y_hat)
    conf = confusion_matrix(y_test, y_hat)
    print("Finished Fold!")
    return f1_macro, f1_micro, f1_weighted, accuracy, conf

#def fold_generator(kfold, data, groups):
#    for i, (train_indices, test_indices) in enumerate(kfold.split(data, groups=dataset_names)):
#        yield (i, train_indices, test_indices)    
COMM = MPI.COMM_WORLD
if (COMM.rank == 0):
    print("Beginning cross validation")
    type_column = 'colType'
    #models = [KNeighborsClassifier,
    #DecisionTreeClassifier,
    #RandomForestClassifier,
    #MLPClassifier,
    #GaussianNB]
    #closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
    closed_embed = 'embedded_d3m_closed.csv'
    #closed_embed_bal = 'closed_data_rebalance.csv'
    print("loading file")
    embed_df = pd.read_csv(closed_embed)
    print("Done loading!")
    #do shuffled cross validation, but that can also be replicated
    X_embed = embed_df.drop(['colType','datasetName'],axis=1).to_numpy()
    print(np.shape(X_embed))
    y = embed_df['colType']
    dataset_names = embed_df['datasetName']
    splitter = LeaveOneGroupOut()
    split_num = splitter.get_n_splits(embed_df, groups=dataset_names)
    jobs = list(splitter.split(embed_df, groups=dataset_names))
    list_jobs_total = [list() for i in range(COMM.size)]
    for i in range(len(jobs)):
        j = i % COMM.size
        list_jobs_total[j].append(jobs[i])
    jobs = list_jobs_total
    print(COMM.size)
else:
    X_embed = None
    y = None
    jobs = None

COMM.Bcast([X_embed, MPI.FLOAT], root=0)
y = COMM.bcast(y,root=0)
jobs = COMM.scatter(jobs, root=0)

results_init = []
for job in jobs:
    train_ind, test_ind = job
    results_init.append(run_fold(train_ind, test_ind))

results_init = MPI.COMM_WORLD.gather(results_init, root = 0)
jobs = list()

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

    print(results)
    results.to_csv(model_name+'_final_cross_val.csv',index=False)
    conf_mean = np.sum(confusions) / len(confusions)
    filename = model_name+'matrix_mean.pkl'
    fileObject = open(filename, 'wb')
    pickle.dump(conf_mean, fileObject)
    fileObject.close()
