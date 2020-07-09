import pandas as pd
import numpy as np
import os.path
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import json
import time
from d3m_profiler.build_table import get_datasets
from BaselineSimon import BaselineSimon
import ModelBase

MAX_LEN = 20
MAX_CELLS = 100
PRIVATE_DATA_DIR = '/users/data/d3m/datasets/training_datasets/'
PRIVATE_METADATA = 'private_d3m_unembed_data.csv.gz'


def evaluate_model(model: ModelBase, use_metadata: bool, n_splits: int, train_size: float=0.66, split_seed=None):
    if use_metadata:
        X, y, groups = parse_metadata()
    else:
        X, y, groups = parse_datasets(get_datasets(PRIVATE_DATA_DIR))
    X, y = model.encode_data(X, y)
    scores = []
    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=split_seed)
    for train_indices, test_indices in splitter.split(X, y, groups):
        start = time.time()
        model.fit(X[train_indices], y[train_indices])
        fold_scores = score(model, X[test_indices], y[test_indices])
        unique, counts = np.unique(y[train_indices], return_counts=True)
        fold_scores['balanced'] = len(np.unique(counts)) == 1
        fold_scores['data_collection'] = model.__class__.__name__
        fold_scores['use_metadata'] = use_metadata
        fold_scores['time_elapsed'] = time.time() - start
        fold_scores['fold_index'] = len(scores)
        scores.append(fold_scores)
        print(scores)
    return scores


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
    data = pd.read_csv(PRIVATE_METADATA).applymap(str)
    X = data[X_labels]
    return X, data['colType'], data['datasetName']


def main():
    results = pd.DataFrame()
    simon = BaselineSimon(max_cells=MAX_CELLS, max_len=MAX_LEN)
    results = results.append(evaluate_model(model=simon, use_metadata=False, split_seed=42, n_splits=9, train_size=0.66), ignore_index=True)
    print(results)
    results.to_csv('experiment_results.csv', index=False)


if __name__ == '__main__':
    main()
