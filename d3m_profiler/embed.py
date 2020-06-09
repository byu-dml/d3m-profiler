import multiprocessing as mp
import pickle as pk
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

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
def embed(df: pd.DataFrame, type_column: str, model_weights_path: str) -> pd.DataFrame:
    model, emb_size = initialize_model(model_weights_path)

    dataset_names = df['datasetName']
    dataset_name_embs = model.encode(df['datasetName'].to_numpy())
    description_embs = model.encode(df['description'].to_numpy())
    col_name_embs = model.encode(df['colName'].to_numpy())
    col_types = df[type_column]

    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(3*len(col_name_embs[0]))])

    return pd.concat([group_type_df, embeddings_df], axis=1)

