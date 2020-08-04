from d3m_profiler import evaluate_models
from models.MetaDataProfiler import MetaDataProfiler
from models.BaselineSimon import BaselineSimon
from models.MetaSimon import MetaSimon
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

#====================
#this is an example of a SIMON and d3m-profiler RandomForestClassifier model
#====================

#the file that contains the unembedded data
#data_path = '../data_files/data/sample_meta.csv'
data_path = '../data_files/data/private_d3m_unembed_metadata.csv'
weights_path = '../data_files/SentenceTransformer'
embed_path ='SentenceTransformerEmbedding.csv'
file_to_save = 'rf_mlp_unbal_4.csv'

model_meta_forest = RandomForestClassifier(random_state=15)
split_type = GroupShuffleSplit(n_splits=4, test_size=0.3, random_state=21)
model_meta_mlp = MLPClassifier(random_state=13)
model_rf = MetaDataProfiler(model=model_meta_forest, use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH=weights_path, model_name='RandomForest_4_Unbal', balance_type='SMOTE', balance=False, embed_data_file=embed_path, split_type=split_type, data_path=data_path)
model_mlp = MetaDataProfiler(model=model_meta_mlp, use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH=weights_path, model_name='MLP_4_Unbal', balance_type='SMOTE', balance=False, embed_data_file=embed_path, split_type=split_type, data_path=data_path)

#get both the models to run with the profiler
initialized_models = [model_rf, model_mlp]
#calling this will save all of the initialized model score results to the save_results_file
evaluate_models.run_models(initialized_models=initialized_models, save_results_file=file_to_save)
