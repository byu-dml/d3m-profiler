import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB as GaussianNB
from sklearn.neural_network import MLPClassifier as MLPClassifier
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from d3m_profiler import rebalance, score_results
from d3m_profiler.evaluate_models import run_models, _save_results
from d3m_profiler.embed_data import embed

results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

type_column = 'colType'
model_weights_path = '../data_files/distilbert-base-nli'
closed_d3m_file = '../data_files/data/closed_d3m_data.csv'

files = [closed_d3m_file]

for _file in files:
    data_collection = _file.split('/')[1]
    print(data_collection)

    orig_df = pd.read_csv(_file)
    orig_df = orig_df.applymap(str)

    #dfs = embed(orig_df, type_column, model_weights_path)

    class_counts = orig_df[type_column].value_counts().values
    balanced = len(set(class_counts)) == 1

    if (not balanced):
        print('rebalancing {} data collection'.format(data_collection))
        rebal_df = rebalance.rebalance_SMOTE(orig_df, type_column, 'smote', model_weights_path)
        dfs = rebal_df

    class_counts = dfs[type_column].value_counts().values
    balanced = len(set(class_counts)) == 1
    print(balanced)

    xtrain, xtest, ytrain, ytest = None, None, None, None

    if (balanced):
        X_syn = dfs[dfs['colName'].eq('SYNTHETIC')].drop(['colName', type_column], axis=1)
        y_syn = dfs[dfs['colName'].eq('SYNTHETIC')][type_column]

        X_organ = dfs[dfs['colName'] != 'SYNTHETIC'].drop(['colName', type_column], axis=1)
        y_organ = dfs[dfs['colName'] != 'SYNTHETIC'][type_column]

        xtrain, xtest, ytrain, ytest = train_test_split(X_organ, y_organ,stratify=y_organ, test_size=0.33)
        print(X_organ)
        print(X_syn)
        xtrain = xtrain.append(X_syn)
        ytrain = ytrain.append(y_syn)
    else:
        X = dfs.drop(['colName', type_column], axis=1)
        y = dfs[type_column]
        column_names = dfs['colName']
            
        xtrain, xtest, ytrain, ytest = train_test_split(X,y,stratify=y,test_size=0.33)

    for model_class in [MLPClassifier,RandomForestClassifier,GaussianNB]:
        classifier = model_class.__name__
        print('evaluating model: {}'.format(classifier))
        model = model_class()
        print('fitting model...')
        model.fit(xtrain, ytrain)
        if (balanced):
            filename = model_class.__name__+'_public_model.sav'
            pickle.dump(model, open(filename, 'wb'))
        yhat = model.predict(xtest)

        accuracy = accuracy_score(ytest, yhat)
        f1_micro = f1_score(ytest, yhat, average='micro')
        f1_macro = f1_score(ytest, yhat, average='macro')
        f1_weighted = f1_score(ytest, yhat, average='weighted')

        results = results.append({'data_collection': data_collection, 'classifier': classifier, 'balanced': balanced, 'accuracy_score': accuracy, 
            'f1_score_micro': f1_micro, 'f1_score_macro': f1_macro, 'f1_score_weighted': f1_weighted}, ignore_index=True)


    print(results)
    results.to_csv('../result_files/results_training.csv', index=False)
