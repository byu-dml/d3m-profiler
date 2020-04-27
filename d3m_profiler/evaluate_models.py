import os
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.ensemble import RandomForestClassifier

from d3m_profiler.embed import EMBEDDED_DATA_PATH, EMBEDDED_SMALL_DATA_PATH


def load_data(small=True):
    if small:
        data_path = EMBEDDED_SMALL_DATA_PATH
    else:
        data_path = EMBEDDED_DATA_PATH

    with open(data_path, 'r') as f:
        data = pd.read_csv(f)

    dataset_names = data['datasetName']
    y = data['colType']
    X = data.drop(['datasetName', 'colType'], axis=1)

    return dataset_names, X, y


def _group_kfold_generator(X, y, groups, n_folds, *args):
    group_kfold = GroupKFold(n_splits=n_folds)
    for i, (train_indices, test_indices) in enumerate(group_kfold.split(X, y, groups)):
        yield (i, X[train_indices], y[train_indices], X[test_indices], test_indices, *args)


def _run_fold(i, X_train, y_train, X_test, test_indices, model_constructor):
    print('fold {}'.format(i))
    model = model_constructor()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return test_indices, y_hat


def run_model(model_constructor, dataset_names, X, y, n_jobs=None):
    """
    Runs models with grouped leave-one-out cross validation, grouped by dataset_names.

    Returns
    -------
    y_hat: pandas.Series
        The predictions of the model for each value of y when used as the test set in the cross validation splitting.
    """

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


def save_results(save_dir, model_name, dataset_names, X, y, y_hat):
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


def main(args):
    if len(args) == 1:
        n_jobs = int(args[0])
    else:
        n_jobs = None

    use_small_data = True  # change to False to use all data

    print('loading data...')
    dataset_names, X, y = load_data(use_small_data)

    save_dir = 'results{}'.format('_small' if use_small_data else '')

    for model_class in [SupportVectorClassifier, RandomForestClassifier]:
        print('evaluating model {}'.format(model_class))
        y_hat = run_model(model_class, dataset_names, X, y, n_jobs)
        print('{} accuracy: {}'.format(model_class, accuracy_score(y, y_hat)))
        save_results(save_dir, str(model_class).split('.')[-1], dataset_names, X, y, y_hat)


if __name__ == '__main__':
    main(sys.argv[1:])  # number of cores
