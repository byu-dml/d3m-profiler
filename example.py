import numpy as np
import multiprocessing as mp
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from d3m_profiler import rebalance
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

open_d3m_file = '~/data/open_d3m_unembed_data.csv'
closed_d3m_file = '~/data/closed_d3m_unembed_data.csv'

#files = [open_d3m_file]
#files = [open_d3m_file, closed_d3m_file]
files = [closed_d3m_file, open_d3m_file]

def fit(model_class, xtrain, ytrain, xtest, ytest, isBalanced, dump=False):
    print('evaluating model: {}'.format(model_class.__name__))
    model = model_class()
    print('fitting model...')
    model.fit(xtrain, ytrain)

    if (isBalanced and dump):
        filename = 'RF_public_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    yhat = model.predict(xtest)
    accuracy = accuracy_score(ytest, yhat)
    f1_micro = f1_score(ytest, yhat, average='micro')
    f1_macro = f1_score(ytest, yhat, average='macro')
    f1_weighted = f1_score(ytest, yhat, average='weighted')

    return {'data_collection': data_collection, 'classifier': model_class.__name__, 'balanced': isBalanced, 'accuracy_score': accuracy, 'f1_score_micro': f1_micro, 'f1_score_macro': f1_macro, 'f1_score_weighted': f1_weighted}

def split_X_and_y(data: pd.DataFrame):
    X = data.drop(['datasetName', type_column], axis=1)
    y = data[type_column]
    return X, y

for _file in files:
    data_collection = _file.split('/')[2]
    print(data_collection)

    orig_df = pd.read_csv(_file)
    orig_df = orig_df.applymap(str)

    embedded_df = embed(orig_df, type_column, model_weights_path)

    class_counts = orig_df[type_column].value_counts().values
    balanced = len(set(class_counts)) == 1

    train_df, test_df = train_test_split(embedded_df, test_size=0.33)
    xtrain, ytrain = split_X_and_y(train_df)
    xtest, ytest = split_X_and_y(test_df)

    results = results.append(fit(RandomForestClassifier, xtrain, ytrain, xtest, ytest, isBalanced=balanced), ignore_index=True)

    if (not balanced):
        # re-fitting with a balanced training set
        print('rebalancing {} data collection'.format(data_collection))
        train_rebal_df = rebalance.rebalance_SMOTE(train_df, type_column, 'smote', model_weights_path)
        xtrain, ytrain = split_X_and_y(train_rebal_df)
        results = results.append(fit(RandomForestClassifier, xtrain, ytrain, xtest, ytest, isBalanced=True), ignore_index=True)


print(results)
results.to_csv('data/results_2.csv', index=False)
