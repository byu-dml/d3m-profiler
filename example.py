import numpy as np
import pathlib as pl
import pandas as pd
import pickle
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, GroupKFold, ShuffleSplit
from d3m_profiler import evaluate_models
from d3m_profiler import embed
import os
import time

file_data = '../../data_files/data/closed_d3m_data.csv'

#start the embedding
COMM = MPI.COMM_WORLD
if (COMM.rank == 0):
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    embedded_df = embed.embed(df=pd.read_csv(file_data), model_weights_path='../../data_files/distilbert-base-nli', embedding_model='SentenceTransformer', use_col_name_only=True)
    file_data_embed = 'data_embed.csv'
    end = time.time()
    print("Time to embed "+str(np.round(end-start,3)))
    start = time.time()
    embedded_df.to_csv(file_data_embed)
    end = time.time()
    print("Time to save "+str(np.round(end-start,3)))
else:
    file_data_embed = 'data_embed.csv'    
COMM.barrier()
initialized_models = [RandomForestClassifier(random_state=15)]
model_names = ['Random Forest']
split_model = 'GroupShuffleSplit'
evaluate_models.run_models(initialized_models=initialized_models, model_names=model_names, balance=True, use_col_name_only=True, use_metadata=True, csv_file_path=file_data_embed,split_model='GroupShuffleSplit')
