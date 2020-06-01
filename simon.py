import time
import os.path
import json
import numpy as np
import pandas as pd
from Simon import Simon
from Simon.Encoder import Encoder
from Simon.DataGenerator import DataGenerator

from build_table import get_datasets


def main(checkpoint, nb_epoch, batch_size, execution_config):
    maxlen = 20
    max_cells = 100
    p_threshold = 0.3

    checkpoint_dir = "pretrained_models/"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    Categories = []

    # get raw_data and header from our data
    datasets = get_datasets('/users/data/d3m/datasets')
    # dataset_names, raw_data, header = [], [], []
    raw_data, header = [], []
    count_columns = 0
    for dataset_id, dataset_doc_path in datasets.items():

        # if count_columns > 500: # DEBUG
        #     break # DEBUG

        # open the dataset doc to get the column headers
        columns = None
        with open(dataset_doc_path, 'r') as dataset_doc:
            # be sure we're skipping w/ the same logic as in build_table.py
            dataset = json.load(dataset_doc)
            # dataset_name = dataset['about']['datasetName']

            for resource in dataset['dataResources']:
                # if resource['resPath'] != 'tables/learningData.csv':
                #     print(f'dataset name: {dataset_name}; resource path: {resource["resPath"]}')
                #     print(f'dataset doc path: {dataset_doc_path}')
                #     if 'columns' in resource:
                #         print(f'num columns skipped: {len(resource["columns"])}')
                #         columns_skipped += len(resource['columns'])

                # if 'resPath' not in resource or resource['resPath'][-4:] != '.csv':
                #     # print(f'Wrong resPath in a data resource. Dataset ID: {dataset_id}; Resource ID: {resource["resID"]}')
                #     pass
                if 'columns' not in resource:
                    # print(f'No columns found in a data resource. Dataset ID: {dataset_id}; Resource ID: {resource["resID"]}')
                    pass
                elif resource['columns'] is not None:

                    columns = resource['columns']

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
                            if values >= max_cells:
                                break
                        dataset = pd.concat(tables, ignore_index=True)

                    for column in columns:
                        values = list(dataset[column['colName']].values)
                        if len(values) > max_cells:
                            values = [str(v)[:maxlen] for v in values[:max_cells]]
                        else:
                            values = [str(v)[:maxlen] for v in values] + ['' for i in range(max_cells - len(values))]

                        # dataset_names.append(dataset_name)
                        raw_data.append(values)
                        header.append((column['colType'],))
                        count_columns += 1

                        if column['colType'] not in Categories:
                            Categories.append(column['colType'])

    del columns
    del dataset

    print(f'Categories: {Categories}', flush=True)

    # dataset_names = np.asarray(dataset_names)
    raw_data = np.asarray(raw_data)
    header = np.asarray(header)

    # return dataset_names, raw_data, header

    start = time.time()
    print(f'Start time: {start}')

    config = {}
    encoder = Encoder(categories=Categories)
    encoder.process(raw_data, max_cells)

    # encode the data
    X, y = encoder.encode_data(raw_data, header, maxlen)

    max_cells = encoder.cur_max_cells

    Classifier = Simon(encoder=encoder)
    data = Classifier.setup_test_sets(X, y)

    # print('Sample chars in X:{}'.format(X[2, 0:10]))
    # print('y:{}'.format(y[2]))
    # print('a number of items from y:{}'.format(y[:20]))

    # need to know number of fixed categories to create model
    category_count = y.shape[1]
    print('Number of fixed categories is: {}'.format(category_count))

    model = Classifier.generate_model(maxlen, max_cells, category_count)
    
    Classifier.load_weights(checkpoint, config, model, checkpoint_dir)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['binary_accuracy'])
    # start = time.time()
    history = Classifier.train_model(batch_size, checkpoint_dir, model, nb_epoch, data)
    # end = time.time()
    # print("Time for training is %f sec"%(end-start))
    config = { 'encoder' :  encoder,
               'checkpoint' : Classifier.get_best_checkpoint(checkpoint_dir) }
    Classifier.save_config(config, checkpoint_dir)
    Classifier.plot_loss(history) #comment out on docker images...
    
    pred_headers = Classifier.evaluate_model(max_cells, model, data, encoder, p_threshold)

    # pred_headers = Classifier.our_train_and_evaluate(batch_size=batch_size, checkpoint_dir=checkpoint_dir, max_len=maxlen, max_cells=max_cells,
    #         category_count=category_count, nb_epoch=1, X=X, y=y, encoder=encoder, p_threshold=p_threshold)

    end = time.time()
    print(f'End time: {end}')
    print(f'Total time taken: {end-start}')

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
