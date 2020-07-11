import sys
from sentence_transformers import SentenceTransformer
import sent2vec
import numpy as np
import pandas as pd
from mpi4py import MPI
import os
from os import path
import time

"""
Initializes a model based on model weights
"""
def embed(model, data_to_embed: pd.DataFrame):
    print("Starting Embedding")
    embeddings = []
    for i in data_to_embed.columns:
        embedding = model.encode_data(data_to_embed[i].str.lower())
        embeddings.append(embedding)
        print("Finished embedding {}".format(i))          

    return pd.DataFrame(data=np.hstack(tuple(embeddings)), columns=['emb_{}'.format(i) for i in range(len(embeddings)*len(embeddings[0][0]))])


def create_save_embeddings(model, df: pd.DataFrame, y_data, groups):
    #check if the embedded file already exists
    if (not path.exists(model.embed_data_file)):
        start = time.time()
        embedding = embed(model=model, data_to_embed=df[model.X_labels])
        #save the DataFrame   
        pd.concat([pd.DataFrame({'datasetName': groups, 'colType': y_data}), embedding],axis=1).to_csv(model.embed_data_file)
        end = time.time()
        print("Time to embed "+str(np.round(end-start,3)))
    else:
        print("Embedding already exists!")
