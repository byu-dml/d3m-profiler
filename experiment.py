import pandas as pd
import numpy as np
import csv
import os.path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import time
from d3m_profiler.build_table import build_table, TableKeys
from BaselineSimon import BaselineSimon
from MetadataProfiler import MetadataProfiler
import ModelBase


DATASET_NAME = TableKeys.DATASET_NAME.value
DESCRIPTION = TableKeys.DESCRIPTION.value
COLUMN_NAME = TableKeys.COLUMN_NAME.value
COLUMN_TYPE = TableKeys.COLUMN_TYPE.value
COLUMN_DATA = TableKeys.COLUMN_DATA.value
METADATA_X_LABELS = [COLUMN_NAME]

MAX_LEN = 20
MAX_CELLS = 100
PRIVATE_DATA_DIR = '/users/data/d3m/datasets/training_datasets/'
PRIVATE_DATA_FILE = 'private_d3m_unembed_data.pkl.gz'


def run_fold(model: ModelBase, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    fold_scores = score(model, X_test, y_test)
    fold_scores['time_elapsed'] = round(time.time() - start, 2)
    return fold_scores


def score(model: ModelBase, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'fold_index': None,
        'classifier': model.__class__.__name__,
        'accuracy_score': round(accuracy_score(y_test, predictions), 4),
        'f1_score_micro': round(f1_score(y_test, predictions, average='micro'), 4),
        'f1_score_macro': round(f1_score(y_test, predictions, average='macro'), 4),
        'f1_score_weighted': round(f1_score(y_test, predictions, average='weighted'), 4),
        'time_elapsed': None,
        'confusion_matrix': multilabel_confusion_matrix(y_test, predictions, labels=np.unique(y_test)),
    }


def parse_data(force=False):
    if force or not os.path.isfile(PRIVATE_DATA_FILE):
        return build_table(PRIVATE_DATA_DIR, include_data=True, max_cells=MAX_CELLS, max_len=MAX_LEN, write_path=PRIVATE_DATA_FILE)
    return pd.read_pickle(PRIVATE_DATA_FILE)


def index_generator(data_shape, groups, splitter):
    for i, (train_indices, test_indices) in enumerate(splitter.split(X=np.zeros(data_shape), groups=groups)):
        yield i, train_indices, test_indices


def standardize_data(data):
    return pd.DataFrame(list(data)).fillna('').to_numpy()


def append_results(filepath: str, scores: dict):
    with open(filepath, 'a') as file:
        writer = csv.DictWriter(file, scores.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(scores)


def main():
    # parse data
    data = parse_data()
    X_data = standardize_data(data[COLUMN_DATA])
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
    splitter = GroupShuffleSplit(n_splits=2, train_size=0.67, random_state=42)
    for fold_index, train_indices, test_indices in index_generator(data.shape, groups, splitter):
        for model, X, y in [simon_tuple, metadata_profiler_tuple]:
            fold_scores = run_fold(model, X[train_indices], y[train_indices], X[test_indices], y[test_indices])
            fold_scores['fold_index'] = fold_index
            append_results('results_conf_mat.csv', fold_scores)
            fold_scores.pop('confusion_matrix')
            append_results('results.csv', fold_scores)
    print(pd.read_csv('results.csv'))


if __name__ == '__main__':
    main()
