import numpy as np
import multiprocessing as mp
import pathlib as pl
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SupportVectorClassifier

from d3m_profiler import rebalance, score_results
from d3m_profiler.evaluate_models import run_models, _save_results
from d3m_profiler.embed import embed

_NUM_THREADS = mp.cpu_count()

results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])



#closed_bal_file = 'data/closed_d3m_bal.csv'
#closed_unbal_file = 'data/closed_d3m_unbal.csv'

#open_bal_file = 'data/open_d3m_bal.csv'
#open_unbal_file = 'data/open_d3m_unbal.csv'

#files = [closed_unbal_file, closed_bal_file, open_unbal_file, open_bal_file]

type_column = 'colType'
model_weights_path = 'torontobooks_unigrams.bin'

open_d3m_file = 'data/open_d3m_data.csv'
closed_d3m_file = 'data/closed_d3m_data.csv'

files = [open_d3m_file]
#files = [open_d3m_file, closed_d3m_file]
#files = [closed_d3m_file, open_d3m_file]

for _file in files:
    data_collection = _file.split('/')[1]
    print(data_collection)

    orig_df = pd.read_csv(_file)
    orig_df = orig_df.applymap(str)

    dfs = [embed(orig_df, type_column, model_weights_path)]

    class_counts = orig_df[type_column].value_counts().values
    balanced = len(set(class_counts)) == 1

    if (not balanced):
        print('rebalancing {} data collection'.format(data_collection))
        rebal_df = rebalance.rebalance_SMOTE(orig_df, type_column, 'smote', model_weights_path)
        dfs.append(rebal_df)

    for df in dfs:
        class_counts = df[type_column].value_counts().values
        balanced = len(set(class_counts)) == 1
        print(balanced)

        xtrain, xtest, ytrain, ytest = None, None, None, None

        if (balanced):
            X_syn = df[df['datasetName'].eq('SYNTHETIC')].drop(['datasetName', type_column], axis=1)
            y_syn = df[df['datasetName'].eq('SYNTHETIC')][type_column]

            X_organ = df[df['datasetName'] != 'SYNTHETIC'].drop(['datasetName', type_column], axis=1)
            y_organ = df[df['datasetName'] != 'SYNTHETIC'][type_column]

            xtrain, xtest, ytrain, ytest = train_test_split(X_organ, y_organ, test_size=0.33)

            xtrain = xtrain.append(X_syn)
            ytrain = ytrain.append(y_syn)
        else:
            X = df.drop(['datasetName', type_column], axis=1)
            y = df[type_column]
            dataset_names = df['datasetName']
            
            xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)

        #for model_class in [SupportVectorClassifier, RandomForestClassifier]:
        for model_class in [RandomForestClassifier]:
            classifier = model_class.__name__
            print('evaluating model: {}'.format(classifier))
            model = model_class()
            print('fitting model...')
            model.fit(xtrain, ytrain)
            if (balanced):
                filename = 'RF_public_model.sav'
                pickle.dump(model, open(filename, 'wb'))
            yhat = model.predict(xtest)

            accuracy = accuracy_score(ytest, yhat)
            f1_micro = f1_score(ytest, yhat, average='micro')
            f1_macro = f1_score(ytest, yhat, average='macro')
            f1_weighted = f1_score(ytest, yhat, average='weighted')

            results = results.append({'data_collection': data_collection, 'classifier': classifier, 'balanced': balanced, 'accuracy_score': accuracy, 
                'f1_score_micro': f1_micro, 'f1_score_macro': f1_macro, 'f1_score_weighted': f1_weighted}, ignore_index=True)


print(results)
results.to_csv('data/results_2.csv', index=False)
