import sys
from sentence_transformers import SentenceTransformer
import sent2vec
import numpy as np
import pandas as pd
from mpi4py import MPI
import os
from os import path
import time
import pickle


def create_save_embeddings(model, df: pd.DataFrame, y_data, groups):
    #check if the embedded file already exists
    if (not path.exists(model.embed_data_file)):
        start = time.time()
        embedding, y_data = model.encode_data(df, y_data)
        #save the DataFrame   
        if (model.pkl is False):
            pd.concat([pd.DataFrame({'datasetName': groups}), y_data, embedding], axis=1).to_csv(model.embed_data_file)
        else:
            embedding = pd.concat([pd.DataFrame({'datasetName': groups}), y_data, embedding],axis=1)
            
            pickle.dump(embedding, open(model.embed_data_file, "wb" ))
        end = time.time()
        print("Time to embed "+str(np.round(end-start,3)))
    else:
        print("Embedding already exists!")
