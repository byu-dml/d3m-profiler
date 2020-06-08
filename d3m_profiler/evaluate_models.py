import os
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

_NUM_THREADS = (mp.cpu_count() - 1)


"""
Constructs a GroupKFold object to fold and iterate over.

Returns (yield)
-------
tuple(i, X[train_indices], y[train_indices], X[test_indices], test_indices, *args): Tuple(int, pd.DataFrame, pd.DataFrame, pd.DataFrame, list(int), *args)
    Arguments necessary for the _run_fold function call
"""
def _group_kfold_generator(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_folds: int, *args):
    print('group_kfold_generator')
    group_kfold = GroupKFold(n_splits=n_folds)
    for i, (train_indices, test_indices) in enumerate(group_kfold.split(X, y, groups)):
        yield (i, X[train_indices], y[train_indices], X[test_indices], test_indices, *args)

"""
Executes a fold from the kfold group.

Returns
-------
tuple(test_indices, y_hat): Tuple(list(int), pd.Series)
    Predictions from current kfold iteration and corresponding indices
"""
def _run_fold(i: int, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, test_indices: np.ndarray, model_constructor):
    print('fold {}'.format(i))
    model = model_constructor()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return test_indices, y_hat

"""
Runs model with grouped leave-one-out cross validation, grouped by dataset_names.

Returns
-------
y_hat: pandas.Series
    The predictions of the model for each value of y when used as the test set in the cross validation splitting.
"""
def _run_model(model_constructor, col_names: pd.Series, X: pd.DataFrame, y: pd.DataFrame, n_jobs: int=None):
    n_folds = len(col_names.unique())
    print('{} folds'.format(n_folds))

    if n_jobs is None:
        n_jobs = max(mp.cpu_count() - 1, 1)

    mp_pool = mp.Pool(n_jobs)
    results = mp_pool.starmap(_run_fold, _group_kfold_generator(X.values, y.values, col_names.values, n_folds, model_constructor))
    mp_pool.close()
    mp_pool.join()

    y_hat = pd.Series([None] * len(y), dtype=str)
    for (indices, values) in results:
        assert y_hat[indices].isna().all()
        y_hat[indices] = values

    return y_hat

"""
Saves predictions from grouped leave-one-out cross validation, grouped by dataset_names.

Returns
-------
None
"""
def _save_results(save_dir: str, model_name: str, col_names: pd.Series, X: pd.DataFrame, y: pd.Series, y_hat: pd.Series):
    data = pd.DataFrame({
        'colName': col_names.values,
        'colType': y.values,
        'colType_predicted': y_hat.values,
    })

    filename = 'predictions_{}.csv'.format(model_name)
    path = os.path.join(save_dir, filename)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(path, 'w') as f:
        data.to_csv(f)

"""
Runs models with grouped leave-one-out cross validation, grouped by dataset_names.
Saves results in "results" directory.

Returns
-------
None
"""
def run_models(X: pd.DataFrame, y: pd.Series, col_names: pd.Series, type_column: str, classifiers: list):
    save_dir = '../result_files'
    results = pd.DataFrame(columns=['classifier', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])
    columns = ['classifier', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted']

    for model_class in classifiers:
        print('evaluating model: {}'.format(model_class.__name__))
        y_hat = _run_model(model_class, col_names, X, y, _NUM_THREADS)
        #get the scores
        accuracy = accuracy_score(y,y_hat)
        f1_micro = f1_score(y, y_hat, average='micro')
        f1_macro = f1_score(y, y_hat, average='macro')
        f1_weighted = f1_score(y, y_hat, average='weighted')
        results = results.append(pd.DataFrame(data = [[model_class.__name__, accuracy, f1_micro, f1_macro, f1_weighted]],columns = columns),ignore_index=True)
    
    print(results)
    results.to_csv('../result_files/results_all_models.csv', index=False)    
        
