import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle
import sys
from d3m_profiler import rebalance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score


results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

type_column = 'colType'
model = RandomForestClassifier(max_depth=10)

closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
closed_embed = 'embedded_d3m_closed.csv'
closed_embed_bal = 'closed_data_rebalance.csv'
_file = closed_d3m_file


print("loading file")
embed_bal_df = pd.read_csv(closed_embed_bal)
print("Done loading!")

#do shuffled cross validation, but that can also be replicated
splitter = GroupShuffleSplit(n_splits = 2, train_size=0.66, random_state = 31)
f1s = list()
matrices = list()
print("Beginning cross validation")
iterator = 1
X_bal = embed_bal_df.drop(['colType','datasetName'],axis=1)
y_bal = embed_bal_df['colType']
dataset_names = embed_bal_df['datasetName']

#for train_ind, test_ind in splitter.split(embed_bal_df,groups = dataset_names):
def run_fold(i:int, train_ind: np.ndarray, test_ind: np.ndarray):
    #now fit on every fold   
    print("fold_num = "+str(i))
    model.fit(X_bal.iloc[train_ind],y_bal.iloc[train_ind])
    y_hat = model.predict(X_bal.iloc[test_ind])
    y_test = y_bal.iloc[test_ind]
    f1s.append(f1_score(y_test, y_hat, labels = y_bal.iloc[train_ind].unique(), average='macro'))
    matrices.append(confusion_matrix(y_test, y_hat, labels=y_bal.iloc[train_ind].unique()))
    
def fold_generator(kfold, data, groups):
    for i, (train_indices, test_indices) in enumerate(kfold.split(data, groups)):
        yield (i, train_indices, test_indices, *args)    
    
mp_pool = mp.Pool(n_jobs)
results = mp_pool.starmap(run_fold, fold_generator(splitter, embed_bal_df, dataset_names))
mp_pool.close()    

print(np.mean(f1s))
print(matrices)    

