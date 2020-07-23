from d3m_profiler import evaluate_models
from models.MetaDataProfiler import MetaDataProfiler
from models.BaselineSimon import BaselineSimon
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

#====================
#this is an example of a SIMON and d3m-profiler RandomForestClassifier model
#====================

#the file that contains the unembedded data
meta_data_path = '../data_files/data/sample.csv'
simon_path = '../data_files/data/data_sample.pkl'
weights_path = '../data_files/SentenceTransformer'
meta_embed_path = 'SentenceTransformer_embedding_col_True.csv'
simon_embed_path = 'Simon_embed.pkl'
file_to_save = 'rf_v_simon.csv'

model_comp = RandomForestClassifier(random_state=30)
model_comp_1 = MLPClassifier(random_state=21)
#define the split_type
split_type = ShuffleSplit(n_splits=4, test_size=0.3, random_state=31)
#create the models
#simon model
model_SIMON = BaselineSimon(split_type=split_type, embed_data_file=simon_embed_path, data_path=simon_path, model_name='Simon_GroupShuffle_9')
#random forest model
model_random_forest = MetaDataProfiler(model=model_comp, use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH=weights_path, model_name='RandomForest_ColName_GroupShuffle_9', balance_type='SMOTE', balance=False, embed_data_file=meta_embed_path, split_type=split_type, data_path=meta_data_path)
#mlp classifier
model_mlp = MetaDataProfiler(model=model_comp_1, use_col_name_only=True, embedding_type='SentenceTransformer', EMBEDDING_WEIGHTS_PATH=weights_path, model_name='MLP_ColName_GroupShuffle_9', balance_type='SMOTE', balance=False, embed_data_file=meta_embed_path, split_type=split_type, data_path=meta_data_path)

#get both the models to run with the profiler
initialized_models = [model_SIMON, model_random_forest]
#calling this will save all of the initialized model score results to the save_results_file
evaluate_models.run_models(initialized_models=initialized_models, save_results_file=file_to_save)
