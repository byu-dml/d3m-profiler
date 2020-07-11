import pandas as pd
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ShuffleSplit
from d3m_profiler import evaluate_models
from d3m_profiler import embed
from models.MetaDataProfiler import MetaDataProfiler

#the file that contains the unembedded data
file_data = '../data_files/data/sample.csv'

#create the models
random_forest = MetaDataProfiler(use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH='../data_files/distilbert-base-nli', model=RandomForestClassifier(random_state=15), model_name="Random Forest", balance_type='SMOTE', balance=True, embed_data_file='sample_embedding_1.csv')

mlp_classifier = MetaDataProfiler(use_col_name_only=False, embedding_type='sent2vec', EMBEDDING_WEIGHTS_PATH='../data_files/torontobooks_unigrams.bin', model=MLPClassifier(random_state=12), model_name="MLP Classifier", balance_type='SMOTE', balance=True, embed_data_file='sample_embedding_2.csv')


initialized_models = [random_forest, mlp_classifier]
#input one of the module imported above as a string
split_model = 'GroupShuffleSplit'
#calling this will save all of the initialized model score results to a csv called 'models_final_cross_val.csv'
evaluate_models.run_models(initialized_models=initialized_models, data_path=file_data, split_model=split_model)
