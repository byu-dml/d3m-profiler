import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
import sent2vec

from build_table import DATA_PATH


EMBEDDED_DATA_PATH = './embedded_data.csv'
EMBEDDED_SMALL_DATA_PATH = './embedded_data_small.csv'
_NUM_THREADS = mp.cpu_count()


def embed(model_weights_path):
    model = sent2vec.Sent2vecModel()
    model.load_model(model_weights_path)
    emb_size = model.get_emb_size()

    with open(DATA_PATH, 'r') as f:
        data = pd.read_csv(f, na_filter=False, dtype={'datasetName': str, 'description': str, 'colName': str, 'colType': str})

    dataset_names = data['datasetName'].tolist()
    dataset_name_embs = model.embed_sentences(data['datasetName'].str.lower(), num_threads=_NUM_THREADS).tolist()
    description_embs = model.embed_sentences(data['description'].str.lower(), num_threads=_NUM_THREADS).tolist()
    col_name_embs = model.embed_sentences(data['colName'].str.lower(), num_threads=_NUM_THREADS).tolist()
    col_types = data['colType'].tolist()

    embedded_data = pd.DataFrame(
        data=np.hstack((
            np.reshape(dataset_names, (-1, 1)),
            dataset_name_embs,
            description_embs,
            col_name_embs,
            np.reshape(col_types, (-1, 1)),
        )),
        columns=['datasetName'] + ['emb_{}'.format(i) for i in range(3*emb_size)] + ['colType']
    )

    with open(EMBEDDED_DATA_PATH, 'w') as f:
        embedded_data.to_csv(f, index=False)

    with open(EMBEDDED_SMALL_DATA_PATH, 'w') as f:
        embedded_data.sample(n=1000, axis=0).to_csv(f, index=False)


if __name__ == '__main__':
    embed(sys.argv[1])  # model weights path
