import multiprocessing as mp
import pickle as pk
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

    dataset_names = df['datasetName']
    dataset_name_embs = model.embed_sentences(df['datasetName'].str.lower(), num_threads=_NUM_THREADS)
    description_embs = model.embed_sentences(df['description'].str.lower(), num_threads=_NUM_THREADS)
    col_name_embs = model.embed_sentences(df['colName'].str.lower(), num_threads=_NUM_THREADS)
    col_types = df[type_column]

    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(3*emb_size)])

    return pd.concat([group_type_df, embeddings_df], axis=1)

