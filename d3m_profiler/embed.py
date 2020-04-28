import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
import sent2vec

_NUM_THREADS = (mp.cpu_count() - 1)

"""
Initializes a Sent2vecModel based on model weights

Returns
-------
tuple(model, emb_size): Tuple(sent2vec.Sent2vecModel, int)
    The Sent2vecModel model and the embedding size.
"""
def initialize_model(model_weights_path: str) -> (sent2vec.Sent2vecModel, int):
    model = sent2vec.Sent2vecModel()
    model.load_model(model_weights_path)
    emb_size = model.get_emb_size()
    
    return model, emb_size
    
"""
Embeds textual data in a DataFrame based on model weights and retains 
group level of dataset name along with the response variable.

Returns
-------
embedded_data: pandas.DataFrame
    The embedded DataFrame along with the group level dataset name and response
    variable.
"""
def embed(df: pd.DataFrame, type_column: str, model_weights_path: str) -> pd.DataFrame:
    model, emb_size = initialize_model(model_weights_path)
    
    dataset_names = df['datasetName'].tolist()
    dataset_name_embs = model.embed_sentences(df['datasetName'].str.lower(), num_threads=_NUM_THREADS).tolist()
    description_embs = model.embed_sentences(df['description'].str.lower(), num_threads=_NUM_THREADS).tolist()
    col_name_embs = model.embed_sentences(df['colName'].str.lower(), num_threads=_NUM_THREADS).tolist()
    col_types = df[type_column].tolist()
    
    embedded_data = pd.DataFrame(
        data=np.hstack((
            np.reshape(dataset_names, (-1, 1)),
            dataset_name_embs,
            description_embs,
            col_name_embs,
            np.reshape(col_types, (-1, 1)),
        )),
        columns=['datasetName'] + ['emb_{}'.format(i) for i in range(3*emb_size)] + [type_column]
    )
    
    return embedded_data