from d3m_profiler import evaluate_models
from models.MetaDataProfiler import MetaDataProfiler
from models.BaselineSimon import BaselineSimon
from models.MetaSimon import MetaSimon
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
import warnings
import numpy as np
warnings.filterwarnings("ignore")

#====================
#this is an example of a SIMON and d3m-profiler RandomForestClassifier model
#====================
np.random.seed(12)
#the file that contains the unembedded data
#data_path = '../data_files/data/sample_meta.csv'
data_path = '../data_files/data/private_d3m_unembed_metadata.csv'
weights_path = '../data_files/SentenceTransformer'
embed_path ='SentenceTransformerEmbedding.csv'
file_to_save = 'rf_mlp_unbal.csv'

forest_list = [rf(random_state=15), rf(random_state=31), rf(random_state=53), rf(random_state=43), rf(random_state=51)]
mlp_list = [mlp(random_state=16), rf(random_state=35), rf(random_state=57), rf(random_state=41), rf(random_state=59)]
initialized_models = []
for i in range(len(forest_list)):
    split_type = GroupShuffleSplit(n_splits=4, test_size=0.3, random_state=np.random.randint(0,21))
    model_rf = MetaDataProfiler(model=forest_list[i], use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH=weights_path, model_name='RandomForest_{}'.format(i), balance_type='SMOTE', balance=False, embed_data_file=embed_path, split_type=split_type, data_path=data_path)
    model_mlp = MetaDataProfiler(model=mlp_list[i], use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH=weights_path, model_name='MLP_{}'.format(i), balance_type='SMOTE', balance=False, embed_data_file=embed_path, split_type=split_type, data_path=data_path)
    initialized_models.append(model_rf)
    initialized_models.append(model_mlp)

#calling this will save all of the initialized model score results to the save_results_file
evaluate_models.run_models(initialized_models=initialized_models, save_results_file=file_to_save)
