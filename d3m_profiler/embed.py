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
def embed(data_to_embed: pd.DataFrame, model_weights_path: str, embedding_model: str):
    print("Starting Embedding")
    #sentence transformer model
    if (embedding_model == 'SentenceTransformer'):
        model = SentenceTransformer(model_weights_path)
        embeddings = []
        for i in data_to_embed.columns:
            embedding = model.encode(data_to_embed[i].str.lower())
            embeddings.append(embedding)
            print("Finished embedding {}".format(i)) 
              
    #sent2vec model         
    elif (embedding_model == 'sent2vec'):
        model = sent2vec.Sent2vecModel()
        model.load_model(model_weights_path)
        embeddings = []
        for i in data_to_embed.columns:
             embedding = model.embed_sentences(data_to_embed[i].str.lower())
             embeddings.append(embedding)
             print("Finished embedding {} column".format(i))            

    return pd.DataFrame(data=np.hstack(tuple(embeddings)), columns=['emb_{}'.format(i) for i in range(len(embeddings)*len(embeddings[0][0]))])


def create_save_embeddings(df: pd.DataFrame, model_weights_path: str, embedding_model: str, use_col_name_only: bool, path_embedding: str):
    COMM = MPI.COMM_WORLD
    if (COMM.rank == 0):
        #check if the embedded file already exists
        if (not path.exists(path_embedding)):
            start = time.time()
            dataset_names = df['datasetName']
            col_types = df['colType']
            if (use_col_name_only is False):
                embedding = embed(df[['colName','datasetName','description']], model_weights_path, embedding_model)
            else:
                embedding = embed(df[['colName']], model_weights_path, embedding_model)
            #save the DataFrame   
            pd.concat([pd.DataFrame({'datasetName': dataset_names, 'colType': col_types}), embedding],axis=1).to_csv(path_embedding)
            end = time.time()
            print("Time to embed "+str(np.round(end-start,3)))
        else:
            print("Embedding already exists!")
    COMM.barrier()
