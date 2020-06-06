import multiprocessing as mp
import pickle
import sys
from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd

_NUM_THREADS = (mp.cpu_count() - 1)

"""
Initializes the model with the sentence transformer
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
def embed(df: pd.DataFrame, type_column: str, model_weights_path: str) -> pd.DataFrame:
    model = initialize_model(model_weights_path)

    dataset_names = df['datasetName']
    dataset_name_embs = model.encode(df['datasetName'].str.lower())
    description_embs = model.encode(df['description'].str.lower())
    col_name_embs = model.encode(df['colName'].str.lower())
    col_types = df[type_column]
    group_type_df = pd.DataFrame({'colType': col_types})
    embeddings_df = pd.DataFrame(data=pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(3*len(col_name_embs[0]))]))

    return pd.concat([group_type_df, embeddings_df], axis=1)
