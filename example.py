import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle
import sys
from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SupportVectorClassifier
from d3m_profiler import rebalance


results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

type_column = 'colType'
model_weights_path = '../data_files/distilbert-base-nli'
model = RandomForestClassifier(max_depth=10)

closed_d3m_file = '../data_file/sdata/closed_d3m_data.csv'

_file = closed_d3m_file

orig_df = pd.read_csv(_file)
orig_df = orig_df.applymap(str)

class_counts = orig_df[type_column].value_counts().values
balanced = len(set(class_counts)) == 1


if (not balanced):
    print('rebalancing {} data collection'.format(data_collection))
    rebal_df = rebalance.rebalance_SMOTE(orig_df, type_column, 'smote', model_weights_path)
    df = rebal_df

df.to_csv('embedded_d3m_closed.csv',index=False)        

"""        
class_counts = df[type_column].value_counts().values
balanced = len(set(class_counts)) == 1
print(balanced)

X = df.drop(['datasetName', type_column], axis=1, inplace=True)
y = df[type_column]
dataset_names = df['datasetName']

#do shuffled cross validation, but that can also be replicated
splitter = GroupShuffleSplit(n_splits = 2, train_size=0.66, random_state = 31)
f1s = list()
matrices = list()
for train_ind, test_ind in splitter.split(df,groups = dataset_names):
    #now fit on every fold   
    model.fit(X[train_ind],y[train_ind])
    y_hat = model.predict(X[test_ind])
    y_test = y[test_ind]
    f1s.append(f1_score(ytest, yhat, labels = y[train_ind].unique(), average='macro'))
    matrices.append
"""

