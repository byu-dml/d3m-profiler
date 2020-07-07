from multiprocessing import cpu_count
import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from d3m_profiler.rebalance import rebalance_SMOTE as rebalance
from d3m_profiler.embed import embed
from create_data_files import create_metadata_file

NUM_THREADS = cpu_count()
TYPE_COLUMN = 'colType'
X_LABELS = ['datasetName', 'description', 'colName']
MODEL_WEIGHTS_PATH = 'torontobooks_unigrams.bin'
PUBLIC_METADATA = 'public_d3m_unembed_data.csv.gz'
PRIVATE_METADATA = 'private_d3m_unembed_data.csv.gz'
PRIVATE_DATA_DIR = '/users/data/d3m/datasets/training_datasets/'


def experiment(dataset_name, X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    fit(model, X_train, y_train)
    scores = score(model, X_test, y_test)
    scores['data_collection'] = dataset_name
    scores['balanced'] = len(y_train.value_counts().unique()) == 1
    return scores


def fit(model, X_train, y_train, model_dump_filename=None):
    print('fitting model...')
    model.fit(X_train, y_train)
    if model_dump_filename:
        pickle.dump(model, open(model_dump_filename, 'wb'))


def score(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        'classifier': model.__class__.__name__,
        'accuracy_score': accuracy_score(y_test, predictions),
        'f1_score_micro': f1_score(y_test, predictions, average='micro'),
        'f1_score_macro': f1_score(y_test, predictions, average='macro'),
        'f1_score_weighted': f1_score(y_test, predictions, average='weighted')
    }


if __name__ == '__main__':
    if not os.path.isfile(PRIVATE_METADATA):
        create_metadata_file(dataset_dir=PRIVATE_DATA_DIR, filename=PRIVATE_METADATA)

    results = pd.DataFrame(columns=['data_collection', 'classifier', 'balanced', 'accuracy_score',
                                    'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])

    for dataset in [PRIVATE_METADATA, PUBLIC_METADATA]:
        print(dataset)
        data = pd.read_csv(dataset).applymap(str)

        X = embed(data[X_LABELS], MODEL_WEIGHTS_PATH)
        y = data[TYPE_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)
        results = results.append(experiment(dataset, X_train, y_train, X_test, y_test), ignore_index=True)

        if len(y_train.value_counts().unique()) > 1:
            print('rebalancing {} data collection'.format(dataset))
            X_train, y_train = rebalance(X_train, y_train, 'smote')
            results = results.append(experiment(dataset, X_train, y_train, X_test, y_test), ignore_index=True)

    print(results)
    results.to_csv('results.csv', index=False)
