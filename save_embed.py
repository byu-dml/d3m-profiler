import multiprocessing as mp
import pickle as pk
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm

_NUM_THREAD = (mp.cpu_count() - 1)

model_weights_path = '../../data_files/distilbert-base-nli'
closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
open_d3m_file = '../../data_files/data/open_d3m_data.csv'

def initialize_model(model_path):
    model = SentenceTransformer(model_path)
    return model
    
def embed(df: pd.DataFrame, model_weights_path: str):
    model = initialize_model(model_weights_path)

    dataset_names = df['datasetName']
    print("Starting Embedding")
    dataset_name_embs = tqdm(model.encode(df['datasetName'].str.lower().to_numpy()))
    print("done_1!")
    description_embs = model.encode(df['description'].str.lower().to_numpy())
    print("done_2!")
    col_name_embs = model.encode(df['colName'].str.lower().to_numpy())
    print(np.shape(col_name_embs))
    print(np.shape(description_embs))
    col_types = df[type_column]

    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(len(col_name_embs[0]))])

    return pd.concat([group_type_df, embeddings_df], axis=1)
if __name__=="__main__":
    #load the original csv
    print("loading...")
    data_df = pd.read_csv(open_d3m_file)
    #split the dataFrame into equal parts
    print("splitting...")
    splits = np.array_split(data_df, _NUM_THREAD)
    processes = []
    print("starting multiprocess")
    for i in range(0,_NUM_THREAD):
        p = mp.Process(target=embed,args=(splits[i],model_weights_path,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    print(p)
    #df = pd.concat(p)
    #save the results
    #df.to_csv('closed_embed_all.csv',index=False)
    
