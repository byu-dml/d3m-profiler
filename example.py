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
closed_embed = '../../data_files/data/embedded_d3m_closed.csv'
weight_model = SentenceTransformer(model_weights_path)
_file = closed_d3m_file



embedded_df = pd.read_csv(closed_embed)
embedded_df = embedded_df.applymap(str)

class_counts = embedded_df[type_column].value_counts().values
balanced = len(set(class_counts)) == 1
print(balanced)
#df = pd.concat([group_type_df, embeddings_df], axis=1)

X_embed = embedded_df.drop(['colType','datasetName'],axis=1)
y = embedded_df['colType']
       
dataset_names = embedded_df['datasetName']  
#balance the data
print("Balancing data...")
smote = SMOTE(k_neighbors=k_neighbors)
X_bal, y_bal = smote.fit_resample(X_embed,y)


num_synthetic = len(X_bal.index) - len(X_embed.index)
#create the rebalanced dataset
rebalanced_df = pd.DataFrame(data=X_bal)
rebalanced_df['colType'] = y_bal
dataset_names = dataset_names.append(pd.Series(['SYNTHETIC'] * num_synthetic), ignore_index=True)
rebalanced_df['datasetName'] = datasets


#do shuffled cross validation, but that can also be replicated
splitter = GroupShuffleSplit(n_splits = 2, train_size=0.66, random_state = 31)
f1s = list()
matrices = list()
for train_ind, test_ind in splitter.split(rebalanced_df,groups = dataset_names):
    #now fit on every fold   
    model.fit(X_bal[train_ind],y_bal[train_ind])
    y_hat = model.predict(X_bal[test_ind])
    y_test = y_bal[test_ind]
    f1s.append(f1_score(y_test, y_hat, labels = y[train_ind].unique(), average='macro'))
    matrices.append(confusion_matrix(y_test, y_hat, labels=y[train_ind].unique()))
print(np.mean(f1s))
print(matrices)    

