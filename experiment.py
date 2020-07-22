import pandas as pd
import numpy as np
import os.path
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import time
from d3m_profiler.build_table import build_table
from BaselineSimon import BaselineSimon
from MetadataProfiler import MetadataProfiler
import ModelBase


DATASET_NAME = 'datasetName'
DESCRIPTION = 'description'
COLUMN_NAME = 'colName'
COLUMN_TYPE = 'colType'
METADATA_X_LABELS = [COLUMN_NAME]

MAX_LEN = 20
MAX_CELLS = 100
PRIVATE_DATA_DIR = '/users/data/d3m/datasets/training_datasets/'
PRIVATE_DATA_FILE = 'private_d3m_unembed_data.pkl.gz'


def run_fold(model: ModelBase, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    fold_scores = score(model, X_test, y_test)
    fold_scores['time_elapsed'] = time.time() - start
    return fold_scores


def score(model: ModelBase, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        'classifier': model.__class__.__name__,
        'accuracy_score': accuracy_score(y_test, predictions),
        'f1_score_micro': f1_score(y_test, predictions, average='micro'),
        'f1_score_macro': f1_score(y_test, predictions, average='macro'),
        'f1_score_weighted': f1_score(y_test, predictions, average='weighted'),
        'confusion_matrix': multilabel_confusion_matrix(y_test, predictions, labels=np.unique(y_test))
    }


def parse_data(force=False):
    if force or not os.path.isfile(PRIVATE_DATA_FILE):
        build_table(PRIVATE_DATA_DIR, include_data=True, max_cells=MAX_CELLS, max_len=MAX_LEN, write_path=PRIVATE_DATA_FILE)
    return pd.read_pickle(PRIVATE_DATA_FILE)


def index_generator(data_shape, groups, splitter):
    for i, (train_indices, test_indices) in enumerate(splitter.split(X=np.zeros(data_shape), groups=groups)):
        yield i, train_indices, test_indices


def standardize_data(data):
    return pd.DataFrame(list(data)).fillna('').to_numpy()


def main():
    # parse data
    data = parse_data()
    X_data = standardize_data(data['data'])
    X_metadata = data[METADATA_X_LABELS]
    y = data[COLUMN_TYPE].to_numpy().reshape(-1, 1)
    groups = data[DATASET_NAME]

    # initialize models
    metadata_profiler = MetadataProfiler()
    simon = BaselineSimon(max_cells=MAX_CELLS, max_len=MAX_LEN)

    # encode data
    X_metadata, y_metadata = metadata_profiler.encode_data(X_metadata, y)
    X_data, y_data = simon.encode_data(X_data, y)

    # pack into tuples
    simon_tuple = (simon, X_data, y_data)
    metadata_profiler_tuple = (metadata_profiler, X_metadata.to_numpy(), y_metadata)

    # run folds
    results = pd.DataFrame()
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.67, random_state=42)
    for fold_index, train_indices, test_indices in index_generator(data.shape, groups, splitter):
        for model, X, y in [simon_tuple, metadata_profiler_tuple]:
            fold_scores = run_fold(model, X[train_indices], y[train_indices], X[test_indices], y[test_indices])
            fold_scores['fold_index'] = fold_index
            results = results.append(fold_scores, ignore_index=True)
    print(results)
    results.to_csv('experiment_results.csv', index=False)


if __name__ == '__main__':
    main()
