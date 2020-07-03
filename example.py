import pandas as pd
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, ShuffleSplit
from d3m_profiler import evaluate_models
from d3m_profiler import embed

#the file that contains the unembedded data
file_data = '../data_files/data/sample.csv'
#this is the file that will contain or already does contain the embedded data
file_data_embed = 'sample_embedding.csv'
#embed the data

#embedding_model can be either 'SentenceTransformer' or 'sent2vec'
embed.embed_data(df=pd.read_csv(file_data), model_weights_path='../data_files/distilbert-base-nli', embedding_model='SentenceTransformer', use_col_name_only=True, path_embedding = file_data_embed)


#now start the testing
initialized_models = [RandomForestClassifier(random_state=15)]
model_names = ['Random Forest']
#input one of the module imported above as a string
split_model = 'GroupShuffleSplit'
evaluate_models.run_models(initialized_models=initialized_models, model_names=model_names, balance=True, csv_file_path=file_data_embed,split_model=split_model)
