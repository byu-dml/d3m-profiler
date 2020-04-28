import os
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.ensemble import RandomForestClassifier

_NUM_THREADS = (mp.cpu_count() - 1)


"""
Constructs a GroupKFold object to fold and iterate over.

Returns (yield)
-------
tuple(i, X[train_indices], y[train_indices], X[test_indices], test_indices, *args): Tuple(int, pd.DataFrame, pd.DataFrame, pd.DataFrame, list(int), *args)
    Arguments necessary for the _run_fold function call
"""
def _group_kfold_generator(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_folds: int, *args):
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
def _run_model(model_constructor, dataset_names: pd.Series, X: pd.DataFrame, y: pd.DataFrame, n_jobs: int=None):
    n_folds = len(dataset_names.unique())
    print('{} folds'.format(n_folds))

    if n_jobs is None:
        n_jobs = max(mp.cpu_count() - 1, 1)

    mp_pool = mp.Pool(n_jobs)
    results = mp_pool.starmap(_run_fold, _group_kfold_generator(X.values, y.values, dataset_names.values, n_folds, model_constructor))
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
def _save_results(save_dir: str, model_name: str, dataset_names: pd.Series, X: pd.DataFrame, y: pd.Series, y_hat: pd.Series):
    data = pd.DataFrame({
        'datasetName': dataset_names.values,
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
def run_models(X: pd.DataFrame, y: pd.Series, dataset_names: pd.Series, type_column: str):
    save_dir = 'results'

    for model_class in [SupportVectorClassifier, RandomForestClassifier]:
        print('evaluating model: {}'.format(model_class.__name__))
        y_hat = _run_model(model_class, dataset_names, X, y, _NUM_THREADS)
        print('{} accuracy: {}'.format(model_class, accuracy_score(y, y_hat)))
        _save_results(save_dir, model_class.__name__, dataset_names, X, y, y_hat)
