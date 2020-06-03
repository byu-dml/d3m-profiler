import time
import os.path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import sys
sys.path.append("./simon")
from Simon import Simon
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator

from d3m_profiler.build_table import get_datasets

MAX_LEN = 20
MAX_CELLS = 100
P_THRESHOLD = 0.3
CHECKPOINT_DIR = 'simon/Simon/pretrained_models/'
DATASET_DIR = '/users/data/d3m/datasets'

def main(checkpoint, nb_epoch, batch_size, execution_config):
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    raw_data, header, categories = get_raw_data(get_datasets(DATASET_DIR))
    print(f'Categories: {categories}', flush=True)

    start = time.time()

    encoder = Encoder(categories=categories)
    encoder.process(raw_data, MAX_CELLS)
    X, y = encoder.encode_data(raw_data, header, MAX_LEN)
    encoder_max_cells = encoder.cur_max_cells
    category_count = y.shape[1] # need to know number of fixed categories to create model
    print('Number of fixed categories is: {}'.format(category_count))

    classifier = Simon(encoder=encoder)
    data = classifier.setup_test_sets(X, y)
    model = classifier.generate_model(MAX_LEN, encoder_max_cells, category_count)
    classifier.load_weights(checkpoint, {}, model, CHECKPOINT_DIR)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    history = classifier.train_model(batch_size, CHECKPOINT_DIR, model, nb_epoch, data)

    config = { 'encoder' :  encoder,
               'checkpoint' : classifier.get_best_checkpoint(CHECKPOINT_DIR) }
    classifier.save_config(config, CHECKPOINT_DIR)
    classifier.plot_loss(history) # comment out on docker images...

    evaluate_model(model, data)
    print(f'Total time taken: {time.time()-start}')


def evaluate_model(model, data):
    start = time.time()
    scores = model.evaluate(data.X_test, data.y_test, verbose=0)
    end = time.time()
    print("Accuracy: %.2f%% \n Time: {0}s \n Time/example : {1}s/ex".format(
        end - start, (end - start) / data.X_test.shape[0]) % (scores[1] * 100))
    
    probabilities = model.predict(data.X_test, verbose=1)
    prediction_indices = probabilities > P_THRESHOLD
    y_pred = np.zeros(data.y_test.shape)
    y_pred[prediction_indices] = 1
    print(f'F1 micro: {f1_score(data.y_test, y_pred, average="micro")}')
    print(f'F1 macro: {f1_score(data.y_test, y_pred, average="macro")}')
    print(f'F1 weighted: {f1_score(data.y_test, y_pred, average="weighted")}')


def get_raw_data(datasets):
    raw_data, header, categories = [], [], []
    count_columns = 0
    for dataset_id, dataset_doc_path in datasets.items():
        # open the dataset doc to get the column headers
        columns = None
        with open(dataset_doc_path, 'r') as dataset_doc:
            dataset = json.load(dataset_doc)
            for resource in dataset['dataResources']:
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
                    count_columns += 1

                    if column['colType'] not in categories:
                        categories.append(column['colType'])
    return np.asarray(raw_data), np.asarray(header), categories


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='attempts to discern data types looking at columns holistically.')

    parser.add_argument('--cp', dest='checkpoint',
                        help='checkpoint to load')

    parser.add_argument('--config', dest='execution_config',
                        help='execution configuration to load. contains max_cells, and encoder config.')

    parser.add_argument('--nb_epoch', dest='nb_epoch', action="store", type=int,
                        default=5, help='number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', action="store", type=int,
                        default=64, help='batch size for training')

    args = parser.parse_args()

    main(args.checkpoint, args.nb_epoch, args.batch_size, args.execution_config)
