import numpy as np
import pathlib as pl
import pandas as pd
import pickle
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from d3m_profiler import evaluate_models
from d3m_profiler import embed

file_data = '../data_files/data/closed_d3m_data.csv'

#start the embedding
COMM = MPI.COMM_WORLD
if (COMM.rank == 0):
    embedded_df = embed.embed(df=pd.read_csv(file_data), model_weights_path='../data_files/distilbert-base-nli-stsb-mean-tokens', embedding_model='SentenceTransformer', use_col_name_only=True)
    file_data_embed = 'data_embed.csv'
    embedded_df.to_csv(file_data_embed)
else:
    file_data_embed = 'data_embed.csv'

COMM.barrier()
initialized_models = [RandomForestClassifier(random_state=15),SupportVectorClassifier()]
model_names = ['Random Forest', 'SVC']
evaluate_models.run_models(initialized_models=initialized_models, model_names=model_names, balance=True, use_col_name_only=True, use_metadata=True, csv_file_path=file_data_embed)
