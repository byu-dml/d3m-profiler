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
def embed(X: pd.DataFrame, model_weights_path: str) -> pd.DataFrame:
    model, emb_size = initialize_model(model_weights_path)

    embeddings = []
    for col in X.columns:
        embedding = model.embed_sentences(X[col].str.lower(), num_threads=_NUM_THREADS)
        embeddings.append(embedding)
    return pd.DataFrame(data=np.hstack(tuple(embeddings)), columns=[f'emb_{i}' for i in range(len(embeddings)*emb_size)])

