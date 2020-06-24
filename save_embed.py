import multiprocessing as mp
import pickle as pk
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

_NUM_THREAD = (mp.cpu_count() - 1)

model_weights_path = '../../data_files/distilbert-base-nli'
closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
open_d3m_file = '../../data_files/data/open_d3m_data.csv'

def initialize_model(model_path):
    model = SentenceTransformer(model_path)
    return model
    
def embed(df: pd.DataFrame, model_weights_path: str):
    model = initialize_model(model_weights_path)
    index = df.index.to_list()
    dataset_names = df['datasetName']
    col_types = df['colType']
    print("Starting Embedding")
    #dataset_name_embs = model.encode(df['datasetName'].to_numpy())
    #print("done_1!")
    #description_embs = model.encode(df['description'].to_numpy())
    #print("done_2!")
    col_name_embs = model.encode(df['colName'].str.lower().to_numpy())
    #print(np.shape(col_name_embs))
    #print(np.shape(description_embs))

    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=col_name_embs, columns=['emb_{}'.format(i) for i in range(len(col_name_embs[0]))],index=index)

    return pd.concat([group_type_df, embeddings_df], axis=1)

if __name__=="__main__":
    #load the original csv
    print("loading...")
    data_df = pd.read_csv(closed_d3m_file).applymap(str)
    #data_df = data_df.sample(100)
    #split the dataFrame into equal parts
    print("splitting...")
    splits = np.array_split(data_df, _NUM_THREAD)
    list_in = list()
    for i in range(len(splits)):
        list_in.append((splits[i],model_weights_path))
    print("starting multiprocess")
    mp_pool = mp.Pool()
    results = mp_pool.starmap(embed, list_in)
    mp_pool.close()
    mp_pool.join()
    
    df = pd.concat(results,axis=0)
    #save the results
    df.to_csv('closed_embed_all_lower.csv',index=False)
    
