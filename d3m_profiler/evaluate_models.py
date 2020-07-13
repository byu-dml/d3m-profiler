import os
import sys
import numpy as np
import pandas as pd
from d3m_profiler import rebalance
from d3m_profiler import embed
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, GroupKFold, ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import pickle
from mpi4py import MPI
"""
Executes a fold from the group of folds.

Returns
-------
tuple(y_hat,y_test): Tuple(list(str), list(str))
    Predictions from current kfold iteration and corresponding indices
"""
def fit_predict_model(X_train, y_train, X_test, y_test, model, rank=None, iter_num = None):
    model.fit(X_train,y_train)
    del X_train
    del y_train
    #predict on the model
    y_hat = list(model.predict(X_test))
    y_test = list(y_test)
    if (rank is not None):
        print("Finished fold {} on processor {}".format(iter_num+1,rank))
    return y_hat, y_test
    
    
"""
Gets the embedded data from csv file

Returns
------

"""
def get_from_csv(data_file: str, to_drop: list):
    
    data = pd.read_csv(data_file)
    X_data = data.drop(to_drop,axis=1)
    y = data['colType']
    num_rows = len(y)
    embed_size = len(X_data.iloc[0])
    groups = data['datasetName']
    return X_data, y, groups, embed_size, num_rows
    
"""
Compiles the results from different processors onto the root processor

Returns
------
results: pd.DataFrame
"""
def compile_results(model_name:str, results_final: list):
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
    conf_matrix = confusion_matrix(y_test, y_hat) 
    results = pd.DataFrame(data=np.array([[model_name, accuracy, f1_macro, f1_micro, f1_weighted]]), columns=['classifier', 'accuracy_score', 'f1_score_macro', 'f1_score_micro', 'f1_score_weighted'])
    return results
   
"""
Gets the variables to be used in the cross validation analysis, each of these variables
will be able to be accessed on all of the processors

Returns
------
Tuple(X_data, jobs, y)

"""
def get_variables(model, data_path: str):
    COMM = MPI.COMM_WORLD
    if (COMM.rank == 0):
        #first embed the data
        X_data, y, groups, _, _ = get_from_csv(data_file = data_path, to_drop=['colType'])
        embed.create_save_embeddings(model, X_data, y, groups)
        #now get the variables from the csv
        X_data, y, groups, embed_size, data_count = get_from_csv(data_file = model.embed_data_file, to_drop=['colType','datasetName']) 
        X_data = X_data.to_numpy()      
        #format jobs list to split across processors
        jobs = get_jobs_list(X_data = X_data, groups = groups, size=COMM.size, cross_type=model.split_type)
    else:
        embed_size = None
        data_count = None
    embed_size = COMM.bcast(embed_size)
    data_count = COMM.bcast(data_count)    
    COMM.barrier()        
    if (COMM.rank != 0):
        X_data = np.empty((data_count, embed_size), dtype='d')
        jobs = None
        y = None
    #get the values from the root processor
    y = COMM.bcast(y,root=0)
    jobs = COMM.scatter(jobs, root=0)
    COMM.Bcast([X_data, MPI.FLOAT], root=0)
    return X_data, jobs, y, COMM.rank

"""
Gets the list of jobs for the multiprocessor system, this is a list based on the number of
processors and the type of cross validation splitting

Returns
------
list_jobs_total (list)
"""
def get_jobs_list(X_data, groups, size: int, cross_type):
    jobs = list(cross_type.split(X_data, groups=groups))
    list_jobs_total = [list() for i in range(size)]
    for i in range(len(jobs)):
        j = i % size
        list_jobs_total[j].append(jobs[i])
    print("Each processor has at most {} jobs".format(len(list_jobs_total[0])))
    return list_jobs_total

"""
Gathers the cross validation results from all processors

Returns
 - results: (pd.DataFrame) - contain f1 and accuracy scores labeled accordingly
"""    
def gather_results(results_init, model_name, root = 0):
    COMM = MPI.COMM_WORLD    
    results_init = MPI.COMM_WORLD.gather(results_init, root = root)
    #pull all results together and broadcast across processors
    if (COMM.rank == root):
        print("Finished cross validation!")
        results = compile_results(results_final=[_i for temp in results_init for _i in temp], model_name= model_name)
    else:
        results = None
    results = COMM.bcast(results, root=root)
    return results
        
"""
Runs model with grouped leave-one-out cross validation, grouped by dataset_names.

Returns
-------
results: (pd.DataFrame) - contains f1 and accuracy scores labeled accordingly
"""
def evaluate_model(model, data_path: str):   
    #get the variables for cross validation
    X_data, jobs, y, rank = get_variables(model=model, data_path=data_path)
    #run cross-validation on all the different processors
    results_init = []
    for it,job in enumerate(jobs):
        train_ind, test_ind = job
        results_init.append(fit_predict_model(X_data[train_ind], y.iloc[train_ind], X_data[test_ind], y.iloc[test_ind], model, rank=rank, iter_num = it))
    del jobs
    #get the total results of the cross_validation tests
    results = gather_results(results_init, model_name=model.model_name, root=0)
    return results
    
"""
Saves the results to csv file in a structured form

Returns
-------
None
"""
def save_results(results_total, model_name: str):
    COMM = MPI.COMM_WORLD
    if (COMM.rank == 0):
        print("Finished model {}".format(model_name))
        print(results_total)
        #now save all of the results
        results_total.to_csv('models_final_cross_val.csv', index=False)
    COMM.barrier()

"""
Runs models with grouped leave-one-out cross validation, grouped by dataset_names.
Saves results in "results" directory.

Returns
-------
None
"""
def run_models(initialized_models: list, data_path=None):
    results_total = pd.DataFrame(columns=['classifier', 'accuracy_score', 'f1_score_macro', 'f1_score_micro', 'f1_score_weighted'])
    for iter_num, model in enumerate(initialized_models):
        results = evaluate_model(model=model, data_path=data_path)
        results_total = results_total.append(results)
        
    save_results(results_total=results_total, model_name=model.model_name)
        
