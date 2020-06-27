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
X_labels = ['datasetName', 'description', 'colName']
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

for _file in files:
    data_collection = _file.split('/')[2]
    print(data_collection)

    orig_df = pd.read_csv(_file)
    orig_df = orig_df.applymap(str)

    X = orig_df[X_labels]
    y = orig_df[type_column]
    X_embedded = embed(X, model_weights_path)

    balanced = len(set(y.value_counts().values)) == 1

    xtrain, xtest, ytrain, ytest = train_test_split(X_embedded, y, test_size=0.33, random_state=42)

    results = results.append(fit(RandomForestClassifier, xtrain, ytrain, xtest, ytest, isBalanced=balanced), ignore_index=True)

    if not balanced:
        # re-fitting with a balanced training set
        print('rebalancing {} data collection'.format(data_collection))
        xtrain, ytrain = rebalance.rebalance_SMOTE(xtrain, ytrain, 'smote')
        results = results.append(fit(RandomForestClassifier, xtrain, ytrain, xtest, ytest, isBalanced=True), ignore_index=True)


print(results)
print(results[['data_collection', 'f1_score_macro']])
results.to_csv('results.csv', index=False)
