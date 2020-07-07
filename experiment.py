import pandas as pd
import numpy as np
import os.path
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import json
import time
from d3m_profiler.build_table import get_datasets
from BaselineSimon import BaselineSimon
import ModelBase

MAX_LEN = 20
MAX_CELLS = 100
PUBLIC_DATA_DIR = '/users/data/d3m/datasets/training_datasets'
PRIVATE_METADATA = 'private_d3m_unembed_data.csv.gz'


def evaluate_model(model: ModelBase, use_metadata: bool, n_splits: int, train_size: float=0.66, split_seed=None, return_times: bool=False):
    if use_metadata:
        X, y, groups = parse_metadata()
    else:
        X, y, groups = parse_datasets(get_datasets(PUBLIC_DATA_DIR))
    X, y = model.encode_data(X, y)
    scores, times = [], []
    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=split_seed)
    for train_indices, test_indices in splitter.split(X, y, groups):
        start = time.time()
        model.fit(X[train_indices], y[train_indices])
        y_pred = model.predict(X[test_indices])
        scores.append(score(y[test_indices], y_pred))
        times.append(time.time() - start)
    if return_times:
        return np.array(scores), np.array(times)
    return np.array(scores)


def score(y_true, y_pred):
    return [f1_score(y_true, y_pred, average='micro'),
            f1_score(y_true, y_pred, average='macro'),
            f1_score(y_true, y_pred, average='weighted'),
            multilabel_confusion_matrix(y_true, y_pred, labels=np.unique(y_true))]


def parse_datasets(datasets):
    raw_data, header, groups = [], [], []
    for dataset_id, dataset_doc_path in datasets.items():
        # open the dataset doc to get the column headers
        with open(dataset_doc_path, 'r') as dataset_doc:
            meta_dataset = json.load(dataset_doc)
            for resource in meta_dataset['dataResources']:
                if 'columns' not in resource:
                    continue
                # then open the actual dataset table to get column values
                if resource['resPath'][-4:] == '.csv':
                    try:
                        dataset = pd.read_csv(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath']))
                    except:
                        continue
                else:
                    values = 0
                    tables = []
                    for entry in os.scandir(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath'])):
                        tables.append(pd.read_csv(entry.path))
                        values += len(tables[-1])
                        if values >= MAX_CELLS:
                            break
                    dataset = pd.concat(tables, ignore_index=True)

                for column in resource['columns']:
                    values = list(dataset[column['colName']].values)
                    if len(values) > MAX_CELLS:
                        values = [str(v)[:MAX_LEN] for v in values[:MAX_CELLS]]
                    else:
                        values = [str(v)[:MAX_LEN] for v in values] + ['' for i in range(MAX_CELLS - len(values))]

                    raw_data.append(values)
                    header.append((column['colType'],))
                    groups.append(meta_dataset['about']['datasetName'])
    return np.asarray(raw_data), np.asarray(header), np.asarray(groups)


def parse_metadata(X_labels=None):
    if X_labels is None:
        X_labels = ['colName']
    data = pd.read_csv(PRIVATE_METADATA)
    X = data[X_labels]
    for col in X.columns:
        X[col] = X[col].str.lower()
    return X, data['colType'], data['datasetName']


def print_scores(scores, times=None):
    np.set_printoptions(precision=5, suppress=True)
    if times is not None:
        print(f'Average time to complete fold: {np.mean(times)} seconds')
        print(f'Total model evaluation time: {np.sum(times)} seconds')
    print(f'Average F1 Micro: {np.mean(scores[:, 0])}')
    print(f'Average F1 Macro: {np.mean(scores[:, 1])}')
    print(f'Average F1 Weighted: {np.mean(scores[:, 2])}')
    print(f'Average Confusion Matrix: \n{np.mean(scores[:, 3], axis=0)}')


def main():
    results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score',
                                    'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

    simon = BaselineSimon(max_cells=MAX_CELLS, max_len=MAX_LEN)
    scores, times = evaluate_model(model=simon, use_metadata=False, split_seed=42, n_splits=9, train_size=0.66, return_times=True)
    print_scores(scores, times)


if __name__ == '__main__':
    main()