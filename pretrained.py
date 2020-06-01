import numpy as np
import pickle
import pandas as pd
import os
from d3m_profiler.embed_data import embed
from d3m_profiler.evaluate_models import run_models, _save_results
from d3m_profiler.build_table import build_table


#this file is already formatted using the build table function in the profiler
#these file paths work specifically on my computer and need to be the paths corresponding to your file system
input_data_file = "../data_files/data/sample.csv"
#weights for the word embedding
model_weights_path = '../data_files/torontobooks_unigrams.bin'
#open pickled file
loaded_model = pickle.load(open("RF_public_model.sav",'rb'))


#get the csv info from the input data
df_in_data = pd.read_csv(input_data_file)
df_in_data = df_in_data.applymap(str)
#now embed the in_data using the embed function in the profiler, this is unlabelled data
df_embed = embed(df_in_data, model_weights_path)
#get the dataset names and column names
X = df_embed.drop(['datasetName'],axis=1)

#now test the model
#need to get y_hat to a pandas dataset
y_hat = loaded_model.predict(X)

#get the dataset names
dataset_names = df_embed['datasetName']

#the directory to save the predicition
save_dir = "../result_files/Prediction_Results"

#save the results (lines 44 - 55)
data = pd.DataFrame({
        'datasetName': dataset_names.values,
        'colType_predicted': y_hat,
    })

filename = 'predictions_{}.csv'.format("profiler")
path = os.path.join(save_dir, filename)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

with open(path, 'w') as f:
    data.to_csv(f)
        
