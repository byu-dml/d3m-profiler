import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle
import sys
from d3m_profiler import rebalance
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB as GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier as MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

num_threads = (mp.cpu_count() - 1)

results = pd.DataFrame(columns=['classifier', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

type_column = 'colType'
models = [KNeighborsClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    MLPClassifier,
    AdaBoostClassifier,
    GaussianNB]

closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
closed_embed = 'embedded_d3m_closed.csv'
closed_embed_bal = 'closed_data_rebalance.csv'


print("loading file")
embed_df = pd.read_csv(closed_embed)
print("Done loading!")

print(embed_df)
#do shuffled cross validation, but that can also be replicated
f1s = list()
matrices = list()
print("Beginning cross validation")
X_embed = embed_df.drop(['colType','datasetName'],axis=1)
y = embed_df['colType']
dataset_names = embed_df['datasetName']
splitter = GroupShuffleSplit(n_splits = len(embed_df['datasetName'].unique()), train_size=0.66, random_state = 31)

for i in models:
    model = i
    model_name = model.__name__
    print("model_name")
    model = model()
    def run_fold(j, train_ind, test_ind):
        #now fit on every fold 
        X_train_embed = X_embed.iloc[train_ind]
        y_train = y.iloc[train_ind]
        print("Balancing training data...")
        k_neighbors = embedded_df['colType'].value_counts().min()-1
        assert k_neighbors > 0, 'Not enough data to rebalance. Must be more than 1:.'
        smote = SMOTE(k_neighbors=k_neighbors)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_embed,y_train) 
        print("fold_num = "+str(j))
        model.fit(X_train_bal,y_train_bal)
        y_hat = model.predict(X_embed.iloc[test_ind])
        y_test = y.iloc[test_ind]
        f1_macro = f1_score(y_test, y_hat, average='macro')
        f1_micro = f1_score(y_test, y_hat, average='micro')
        f1_weighted = f1_score(y_test, y_hat, average='weighted')
        accuracy = accuracy_score(y_test, y_hat)
        conf = confusion_matrix(y_test, y_hat)
        return f1_macro, f1_micro, f1_weighted, accuracy, conf

    def fold_generator(kfold, data, groups):
        for i, (train_indices, test_indices) in enumerate(kfold.split(data, groups=groups)):
            yield (i, train_indices, test_indices)    
    
    mp_pool = mp.Pool(max(num_threads,1))
    results_cross = mp_pool.starmap(run_fold, fold_generator(splitter, embed_df, dataset_names))
    mp_pool.close()    
    mp_pool.join()
    f1s_macro = list()
    f1s_micro = list()
    f1s_weighted = list()
    accuracys = list()
    for f1_macro, f1_micro, f1_weighted, accuracy, matrix in results_cross:
        f1s_macro.append(f1_macro)
        f1s_micro.append(f1_micro)
        f1s_weighted.append(f1_weighted)
        accuracys.append(accuracy)
        matrices.append(matrix)
        
    mean_f1_macro = np.mean(f1s_macro)    
    mean_f1_micro = np.mean(f1s_micro)
    mean_f1_weighted = np.mean(f1s_weighted)
    mean_accuracy = np.mean(accuracy)     
        
        
    results = results.append({'classifier': model_name, 'accuracy_score': mean_accuracy, 'f1_score_micro': mean_f1_micro, 'f1_score_macro': mean_f1_macro, 'f1_score_weighted': mean_f1_weighted}, ignore_index=True) 

print(results)
results.to_csv('final_cross_val.csv',index-False)
