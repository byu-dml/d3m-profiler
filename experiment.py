import pandas as pd
import numpy as np
import os.path
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import time
from d3m_profiler.build_table import get_datasets, extract_data_values, extract_columns
from BaselineSimon import BaselineSimon
from MetadataProfiler import MetadataProfiler
import ModelBase

MAX_LEN = 20
MAX_CELLS = 100
PRIVATE_DATA_DIR = '/users/data/d3m/datasets/training_datasets/'
PRIVATE_METADATA_FILE = 'private_d3m_unembed_metadata.csv.gz'
PRIVATE_BASELINE_DATA_FILE = 'private_d3m_unembed_baseline_data.pkl.gz'
METADATA_X_LABELS = ['colName']


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


def parse_all_data(force_rebuild=False):
    if force_rebuild or not os.path.isfile(PRIVATE_BASELINE_DATA_FILE):
        data = extract_data_values(get_datasets(PRIVATE_DATA_DIR), max_cells=MAX_CELLS, max_len=MAX_LEN)
        data.to_pickle(PRIVATE_BASELINE_DATA_FILE)
    else:
        data = pd.read_pickle(PRIVATE_BASELINE_DATA_FILE)

    return data['values'], data['colType'], data['datasetName']


def parse_metadata(force_rebuild=False):
    if force_rebuild or not os.path.isfile(PRIVATE_METADATA_FILE):
        data = extract_columns(get_datasets(PRIVATE_DATA_DIR))
        data.to_csv(PRIVATE_METADATA_FILE, index=False)
    else:
        data = pd.read_csv(PRIVATE_METADATA_FILE).applymap(str)
    return data[METADATA_X_LABELS], data['colType'], data['datasetName']


def index_generator(data_shape, groups):
    splitter = GroupShuffleSplit(n_splits=5, train_size=0.67, random_state=42)
    for i, (train_indices, test_indices) in enumerate(splitter.split(X=np.zeros(data_shape), groups=groups)):
        yield i, train_indices, test_indices


def main():
    X_metadata, y_metadata, groups_metadata = parse_metadata()
    metadata_profiler = MetadataProfiler(X_labels=METADATA_X_LABELS)
    X_metadata, y_metadata = metadata_profiler.encode_data(X_metadata, y_metadata)

    X_data, y_data, groups_data = parse_all_data()
    simon = BaselineSimon(max_cells=MAX_CELLS, max_len=MAX_LEN)
    X_data, y_data = simon.encode_data(np.asarray(list(X_data)), list(y_data))

    results = pd.DataFrame()
    for fold_index, train_indices, test_indices in index_generator(X_metadata.shape, groups_metadata):
        fold_scores_simon = run_fold(simon, X_data[train_indices], y_data[train_indices],
                                     X_data[test_indices], y_data[test_indices])
        fold_scores_simon['fold_index'] = fold_index

        fold_scores_profiler = run_fold(metadata_profiler, X_metadata.iloc[train_indices], y_metadata.iloc[train_indices],
                                        X_metadata.iloc[test_indices], y_metadata.iloc[test_indices])
        fold_scores_profiler['fold_index'] = fold_index

        results = results.append([fold_scores_simon, fold_scores_profiler], ignore_index=True)
    print(results)
    results.to_csv('experiment_results.csv', index=False)


if __name__ == '__main__':
    main()
