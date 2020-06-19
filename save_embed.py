import multiprocessing as mp
import pickle as pk
import sys

import numpy as np
import pandas as pd

_NUM_THREAD = (mp.cpu_count() - 1)

model_weights_path = '../data_files/distilbert-base-nli'
closed_d3m_file = '../data_files/data/closed_d3m_data.csv'
open_d3m_file = '../data_files/data/open_d3m_data.csv'

def initialize_model(model_path):
    model = SentenceTransformer(model_path: str):
    return model
    
def embed(df: pd.Dataframe, model_weights_path: str):
    model, emb = initialize_model(model_weights_path)

    dataset_names = df['datasetName']
    dataset_name_embs = model.encode(df['datasetName'].str.lower())
    description_embs = model.encode(df['description'].str.lower())
    col_name_embs = model.encode(df['colName'].str.lower())
    col_types = df[type_column]

    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(len(col_name_embs[0])])

    return pd.concat([group_type_df, embeddings_df], axis=1)

for i in range(_NUM_THREAD):
    #load the original csv
    data_df = pd.read_csv(open_d3m_file)
    #split the dataFrame into equal parts
    splits = np.array_split(data_df, _NUM_THREAD)
    processes = []
    for i in range(0,_NUM_THREAD):
        p = mp.Process(target=embed,args=(splits[i],model_weights_path,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    print(p)
    df = pd.concat(p)
    #save the results
    df.to_csv('closed_embed_all.csv',index=False)
    
