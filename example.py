import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle
import sys
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from d3m_profiler import rebalance, score_results
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from d3m_profiler import rebalance


results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

type_column = 'colType'
model_weights_path = '../../data_files/distilbert-base-nli'
model = RandomForestClassifier(max_depth=10)

closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
weight_model = SentenceTransformer(model_weights_path)
_file = closed_d3m_file

orig_df = pd.read_csv(_file)
orig_df = orig_df.applymap(str)

class_counts = orig_df[type_column].value_counts().values
balanced = len(set(class_counts)) == 1


print("Embedding Data")
print("Embedding datasetName...")
dataset_name_embs = weight_model.encode(orig_df['datasetName'].str.lower())
print("Embedding description...")
description_embs = weight_model.encode(orig_df['description'].str.lower())
print("Embedding column name...")
col_name_embs = weight_model.encode(orig_df['colName'].str.lower())

group_type_df = pd.DataFrame({'datasetName': orig_df['datasetName'], 'colType': orig_df['colType']})
print("Building embedded dataframe...")
embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(3*len(col_name_embs[0]))])
df = pd.concat([group_type_df, embeddings_df], axis=1)
print("Saving Embeddings...")
df.to_csv('embedded_d3m_closed.csv',index=False)        
#balance the data
#print("Balancing data...")
#smote = SMOTE(k_neighbors=k_neighbors)
#X_bal, y_bal = smote.fit_resample(X_embed,y)
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

