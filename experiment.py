import pandas as pd
import numpy as np
import csv
import os
import os.path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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
PRIVATE_DATA_FILE = os.getenv('PRIVATE_DATA_FILE_DIR', '') + 'private_d3m_unembed_data.pkl.gz'


def run_fold(model: ModelBase, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    fold_scores = score(model, X_test, y_test)
    fold_scores['time_elapsed'] = format_time_elapsed(time.time() - start)
    return fold_scores


def format_time_elapsed(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)}:{int(minutes)}:{round(seconds, 2)}'


def score(model: ModelBase, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'fold_index': None,
        'classifier': model.__class__.__name__,
        'note': None,
        'splitter': None,
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


def init_model(model: ModelBase, X_columns, trim_to_index=None):
    data = parse_data()
    if trim_to_index:
        print('Trimming data for testing purposes only')
        data = data[:trim_to_index]
        data = data.drop(index=42)  # drop instance with unique label; it breaks rebalancing

    if X_columns == COLUMN_DATA:
        X = standardize_data(data[X_columns])
    else:
        X = data[X_columns]
    y = data[[COLUMN_TYPE]].to_numpy()

    X, y = model.encode_data(X, y)
    return model, X, y


def get_groups(trim_to_index=None):
    data = parse_data()
    if trim_to_index:
        data = data[:trim_to_index]
        data = data.drop(index=42)
    return data[DATASET_NAME]


def main():
    trim = None
    groups = get_groups(trim_to_index=trim)
    models = [
        init_model(MetadataProfiler(rebalance=True, encoder='sentence_transformer'), [COLUMN_NAME], trim_to_index=trim),
        init_model(MetadataProfiler(rebalance=False, encoder='sentence_transformer'), [COLUMN_NAME], trim_to_index=trim),
        init_model(MetadataProfiler(rebalance=True, encoder='sentence_transformer'), [DATASET_NAME, COLUMN_NAME], trim_to_index=trim),
        init_model(MetadataProfiler(rebalance=False, encoder='sentence_transformer'), [DATASET_NAME, COLUMN_NAME], trim_to_index=trim),

        init_model(MetadataProfiler(rebalance=True, encoder='sentence_transformer'), ['resID', COLUMN_NAME], trim_to_index=trim),
        init_model(MetadataProfiler(rebalance=False, encoder='sentence_transformer'), ['resID', COLUMN_DATA], trim_to_index=trim),

        init_model(BaselineSimon(), COLUMN_DATA, trim_to_index=trim),
    ]

    splitters = [
        GroupShuffleSplit(n_splits=4, train_size=0.67, random_state=42),
    ]
    print('Beginning Training')
    for splitter in splitters:
        for fold_index, train_indices, test_indices in index_generator(groups.shape, groups, splitter):
            for model, X, y in models:
            fold_scores = run_fold(model, X[train_indices], y[train_indices], X[test_indices], y[test_indices])
            fold_scores['fold_index'] = fold_index
            fold_scores['note'] = model.get_note()
                fold_scores['splitter'] = splitter.__class__.__name__
                append_results('results_conf_mat2.csv', fold_scores)
            fold_scores.pop('confusion_matrix')
                append_results('results2.csv', fold_scores)
    print(pd.read_csv('results.csv'))


def stacking():
    trim = None
    groups = get_groups(trim_to_index=trim)
    metadata_model, meta_x, meta_y = init_model(MetadataProfiler(rebalance=True, encoder='sentence_transformer'), [DATASET_NAME, COLUMN_NAME], trim_to_index=trim)
    simon_model, data_x, data_y = init_model(BaselineSimon(), COLUMN_DATA, trim_to_index=trim)

    splitter = GroupShuffleSplit(n_splits=1, train_size=0.67, random_state=42)
    metadata_scores, simon_scores, actual = np.array([]), np.array([]), np.array([])
    for fold_index, train_indices, test_indices in index_generator(groups.shape, groups, splitter):
        metadata_model.fit(meta_x[train_indices], meta_y[train_indices])
        simon_model.fit(data_x[train_indices], data_y[train_indices])
        metadata_scores = np.append(metadata_scores, metadata_model.predict(meta_x[test_indices]))
        simon_scores = np.append(simon_scores, simon_model.predict(data_x[test_indices]))
        actual = np.append(actual, meta_y[test_indices])
    
    one_hot = OneHotEncoder()
    one_hot.fit(np.unique(actual).reshape(-1, 1))
    ordinal = OrdinalEncoder()
    ordinal.fit(np.unique(actual).reshape(-1, 1))
    X = np.hstack((one_hot.transform(metadata_scores.reshape(-1, 1)).toarray(), one_hot.transform(simon_scores.reshape(-1, 1)).toarray()))
    y = ordinal.transform(actual.reshape(-1, 1))
    
    splitter = ShuffleSplit(n_splits=4, train_size=0.67, random_state=42)
    for train_indices, test_indices in splitter.split(X):
        lr = LogisticRegression(random_state=42).fit(X[train_indices], y[train_indices])
        preds = lr.predict(X[test_indices])
        scores = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'accuracy_score': round(accuracy_score(preds, y[test_indices]), 4),
            'f1_score_micro': round(f1_score(preds, y[test_indices], average='micro'), 4),
            'f1_score_macro': round(f1_score(preds, y[test_indices], average='macro'), 4),
            'f1_score_weighted': round(f1_score(preds, y[test_indices], average='weighted'), 4),
        }
        append_results('stacking.csv', scores)


if __name__ == '__main__':
    main()
