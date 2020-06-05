import multiprocessing as mp
import pickle
import sys
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

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
def initialize_model(model_weights_path: str):
    model = SentenceTransformer(model_weights_path) 
       
    return model
    
"""
Embeds textual data in a DataFrame based on model weights and retains 
group level of dataset name along with the response variable.

Returns
-------
embedded_data: pandas.DataFrame
    The embedded DataFrame along with the group level dataset name and response
    variable.
"""
def embed(df: pd.DataFrame, model_weights_path: str) -> pd.DataFrame:
    model = initialize_model(model_weights_path)

    dataset_names = df['datasetName']
    dataset_name_embs = model.encode([df['datasetName'].lower()])
    print(np.shape(dataset_name_embs))
    description_embs = model.encode([df['description'].lower()])
    print(np.shape(description_embs))
    col_name_embs = model.encode([df['colName'].lower()])
    print(np.shape(col_name_embs))
    embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(3*len(dataset_name_embs[0]))])

    return embeddings_df
