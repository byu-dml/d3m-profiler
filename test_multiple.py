import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle

from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB as GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier as MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
from sklearn.svm import SVC as SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from d3m_profiler import rebalance, score_results
from d3m_profiler.evaluate_models import run_models, _save_results
from d3m_profiler.embed_data import embed

classifiers = [KNeighborsClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    MLPClassifier,
    AdaBoostClassifier,
    GaussianNB]

type_column = 'colType'
model_weights_path = '../data_files/distilbert-base-nli'

closed_d3m_file = '../data_files/data/closed_d3m_data.csv'
#closed_d3m_file = '../data_files/data/sample.csv'

#get the files to loop through
files = [closed_d3m_file]

for _file in files:
    data_collection = _file.split('/')[1]

    orig_df = pd.read_csv(_file)
    orig_df = orig_df.applymap(str)
    #get a random smaller sample from orig_df so that run_models will work without overflow
    orig_df = orig_df.sample(frac=0.20).reset_index()
    orig_df = orig_df.drop(['datasetName','description'],axis=1)

    class_counts = orig_df[type_column].value_counts().values
    balanced = len(set(class_counts)) == 1
 
    print('rebalancing {} data collection'.format(data_collection))
    dfs = rebalance.rebalance_SMOTE(orig_df, type_column, 'smote', model_weights_path)

    class_counts = dfs[type_column].value_counts().values
    balanced = len(set(class_counts)) == 1
    print(balanced)

    #get the needed info to run all the models
    X = dfs.drop([type_column,'colName'], axis=1)
    y = dfs[type_column]
    column_names = dfs['colName']
        
    #run models on data to determine the best classification model
    run_models(X,y,column_names,type_column,classifiers)
    
