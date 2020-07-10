import os
import sys
import numpy as np
import pandas as pd
from d3m_profiler import rebalance
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
def fit_predict_model(X_train, y_train, X_test, y_test, model, balance=True, rank=None, iter_num = None):
    #now fit using the indeces given by the kfold splitter
    if (balance == True):
        X_train, y_train = rebalance.rebalance_SMOTE(X_train, y_train, 'SMOTE')     
    #fit on  data
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
def get_from_csv(data_file: str):
    
    data = pd.read_csv(data_file)
    X_data = data.drop(['colType','datasetName'],axis=1).to_numpy()
    y = data['colType']
    num_rows = len(y)
    embed_size = len(X_data[0])
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
def get_variables(data_csv_path, num_processors: int, split_model: str, COMM):
    if (COMM.rank == 0):
        X_data, y, groups, embed_size, data_count = get_from_csv(data_file = data_csv_path)       
        #format jobs list to split across processors
        jobs = get_jobs_list(X_data = X_data, groups = groups, size=num_processors, cross_type=split_model)
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
    
    return X_data, jobs, y

"""
Gets the list of jobs for the multiprocessor system, this is a list based on the number of
processors and the type of cross validation splitting

Returns
------
list_jobs_total (list)
"""
def get_jobs_list(X_data, groups, size: int, cross_type):
    if (cross_type=='LeaveOneGroupOut'):
        splitter = LeaveOneGroupOut()
    elif(cross_type=='GroupShuffleSplit'):
        splitter = GroupShuffleSplit(n_splits=9, train_size=0.7, random_state=12)
    elif(cross_type=='ShuffleSplit'):
        splitter = ShuffleSplit(n_splits=1, train_size=0.7, random_state=36)
    else:
        raise ValueError("Type of splitting not available")
    jobs = list(splitter.split(X_data, groups=groups))
    list_jobs_total = [list() for i in range(size)]
    for i in range(len(jobs)):
        j = i % size
        list_jobs_total[j].append(jobs[i])
    print("Each processor has at most {} jobs".format(len(list_jobs_total[0])))
    return list_jobs_total
    
"""
Runs model with grouped leave-one-out cross validation, grouped by dataset_names.

Returns
-------
results: (pd.DataFrame) - contains f1 and accuracy scores labeled accordingly
"""
def evaluate_model(balance: bool, model_name: str, model, data_csv_path, split_model: str):   
    #start the MPI
    COMM = MPI.COMM_WORLD
    #get the variables for cross validation
    X_data, jobs, y = get_variables(data_csv_path=data_csv_path, num_processors=COMM.size, split_model=split_model,COMM=COMM)
    #get the values from the root processor
    y = COMM.bcast(y,root=0)
    jobs = COMM.scatter(jobs, root=0)
    COMM.Bcast([X_data, MPI.FLOAT], root=0)
    #run cross-validation on all the different processors
    results_init = []
    for it,job in enumerate(jobs):
        train_ind, test_ind = job
        results_init.append(fit_predict_model(X_data[train_ind], y.iloc[train_ind], X_data[test_ind], y.iloc[test_ind], model, balance=balance, rank=COMM.rank, iter_num = it))
    #gather results from processors
    results_init = MPI.COMM_WORLD.gather(results_init, root = 0)
    del jobs
    #compile and save the results
    if (COMM.rank == 0):
        del X_data
        del y
        print("Finished cross validation!")
        results = compile_results(results_final = [_i for temp in results_init for _i in temp], model_name = model_name)
    else:
        results = None
    results = COMM.bcast(results,root=0)
    return results

"""
Saves predictions from grouped leave-one-out cross validation, grouped by dataset_names.

Returns
-------
None
"""
def _save_results(save_dir: str, model_name: str, dataset_names: pd.Series, X: pd.DataFrame, y: pd.Series, y_hat: pd.Series):
    data = pd.DataFrame({
        'datasetName': dataset_names.values,
        'colType': y.values,
        'colType_predicted': y_hat.values,
    })

    filename = 'predictions_{}.csv'.format(model_name)
    path = os.path.join(save_dir, filename)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(path, 'w') as f:
        data.to_csv(f)

"""
Runs models with grouped leave-one-out cross validation, grouped by dataset_names.
Saves results in "results" directory.

Returns
-------
None
"""
def run_models(initialized_models: list, model_names: list, balance: bool, csv_file_path=None, split_model='LeaveOneGroupOut'):
    results_total = pd.DataFrame(columns=['classifier', 'accuracy_score', 'f1_score_macro', 'f1_score_micro', 'f1_score_weighted'])
    for iter_num, model in enumerate(initialized_models):
        results = evaluate_model(balance=balance, model_name=model_names[iter_num], model=model, data_csv_path=csv_file_path,split_model=split_model)
        COMM = MPI.COMM_WORLD
        if (COMM.rank == 0):
            print("Finished model {}".format(model_names[iter_num]))
            print(results)
            results_total = results_total.append(results)
            #now save all of the results        
            results_total.to_csv('models_final_cross_val.csv',index=False)
        COMM.barrier()
        
