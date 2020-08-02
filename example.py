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
data_path = '../data_files/data/total_data_sample.pkl'
weights_path = '../data_files/SentenceTransformer'
embed_path ='embedding_both.pkl'
file_to_save = 'rf_simon_mlp.csv'

model_meta = RandomForestClassifier(random_state=15)
model_both = MLPClassifier(random_state=12)
split_type = GroupShuffleSplit(n_splits=4, test_size=0.3, random_state=21)
model_combined = MetaSimon(data_path=data_path, embed_data_path=embed_path, model_meta=model_meta, model_both=model_both, balance=False, split_type=split_type)

#get both the models to run with the profiler
initialized_models = [model_combined]
#calling this will save all of the initialized model score results to the save_results_file
evaluate_models.run_models(initialized_models=initialized_models, save_results_file=file_to_save)
