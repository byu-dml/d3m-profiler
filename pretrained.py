import numpy as np
import pickle
import pandas as pd
import os
from d3m_profiler.embed_data import embed
from d3m_profiler.evaluate_models import run_models, _save_results
from d3m_profiler.build_table import build_table
from sklearn.metrics import accuracy_score, f1_score


#this file is already formatted using the build table function in the profiler

#these file paths work specifically on my computer and need to be the paths corresponding to your file system
input_data_file = "../data_files/data/open_data.csv"
#input_data_file = "../data_files/data/sample.csv"
#weights for the word embedding
model_weights_path = '../data_files/distilbert-base-nli'

#save and the load the needed things
results = pd.DataFrame(columns=['model', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted'])
columns = ['model', 'accuracy_score', 'f1_score_micro', 'f1_score_macro', 'f1_score_weighted']

models = ['GaussianNB_public_model.sav','MLPClassifer_public_model.sav','RandomForestClassifier_public_model.sav']
#get the csv info from the input data
df_in_data = pd.read_csv(input_data_file)
df_in_data = df_in_data.applymap(str)
y = df_in_data['colType']
#now embed the in_data using the embed function in the profiler, this is unlabelled data
df_embed = embed(df_in_data,'colType', model_weights_path)
#get the dataset names and column names
X = df_embed.drop(['colName','colType'],axis=1)

for i in models:
    #open pickled file
    loaded_model = pickle.load(open(i,'rb'))

    #now test the model
    #need to get y_hat to a pandas dataset
    y_hat = loaded_model.predict(X)
    
    results_prediction = pd.DataFrame({'colName':df_in_data['colName'], 'real':df_in_data['colType'], 'predicted':y_hat})
    #get the scores
    accuracy = accuracy_score(y,y_hat)
    f1_micro = f1_score(y, y_hat, average='micro')
    f1_macro = f1_score(y, y_hat, average='macro')
    f1_weighted = f1_score(y, y_hat, average='weighted')
    results = results.append(pd.DataFrame(data = [[i, np.round(accuracy,5), np.round(f1_micro,5), np.round(f1_macro, 5), np.round(f1_weighted,5)]],columns = columns),ignore_index=True)
    results_prediction.to_csv('../result_files/results_'+i+'.csv',index=False)
    
print(results)
results.to_csv('../result_files/results_pretrained_models.csv', index=False)    
