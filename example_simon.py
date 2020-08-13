from d3m_profiler import evaluate_models
from models.MetaDataProfiler import MetaDataProfiler
from models.BaselineSimon import BaselineSimon
from models.MetaSimon import MetaSimon
from models.SimonMLP import SimonMLP
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#the file that contains the unembedded data
data_path = '../data_files/data/private_d3m_unembed_baseline_data.pkl'
#data_path = '../data_files/data/sample_data.pkl'
embed_path ='simon_embed.pkl'
file_to_save = 'simon_test_0.csv'
np.random.seed(12)
split_type = GroupShuffleSplit(n_splits=4, test_size=0.3, random_state=np.random.randint(25,size=5)[0])
#np.random.seed(15)
#model_simon_mlp = SimonMLP(split_type=split_type, embed_data_file=embed_path, data_path=data_path, seed=np.random.randint(25,size=5)[0])
np.random.seed(15)
model_simon = BaselineSimon(split_type=split_type, embed_data_file=embed_path, data_path=data_path, seed=np.random.randint(25,size=5)[0])


#get both the models to run with the profiler
initialized_models = [model_simon]
#calling this will save all of the initialized model score results to the save_results_file
evaluate_models.run_models(initialized_models=initialized_models, save_results_file=file_to_save)
