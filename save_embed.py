import multiprocessing as mp
import pickle as pk
import sys
from sentence_transformers import SentenceTransformer
import sent2vec
import numpy as np
import pandas as pd

model_weights_path_st = '../../data_files/distilbert-base-nli'
model_weights_path_sv = '../../data_files/'
closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
open_d3m_file = '../../data_files/data/open_d3m_data.csv'
    
def embed(df: pd.DataFrame, model_weights_path: str, embedding_model, use_col_name_only: bool):
    index = df.index.to_list()
    dataset_names = df['datasetName']
    col_types = df['colType']
    print("Starting Embedding")
    if (embedding_model = 'SentenceTransformer'):
        model = SentenceTransformer(model_weights_path)
        col_name_embs = model.encode(df['colName'].str.lower().to_numpy())
        all_embeddings = col_name_embs
        if (use_col_name_only is False):
            dataset_name_embs = model.encode(df['datasetName'].str.lower().to_numpy())
            print("Finished dataset names!")
            description_embs = model.encode(df['description'].str.lower().to_numpy())
            print("Finished descriptions!")
            all_embeddings = np.hstack((dataset_name_embs, description_embs, col_name_embs))
        
    elif (embedding_model = 'sent2vec'):
        model = sent2vec.Sent2vecModel()
        model.load_model(model_weights_path)
        col_name_embs = model.embed_sentences(df['colName'].str.lower())
        all_embeddings = col_name_embs
        if (use_col_name_only is False):
            dataset_name_embs = model.embed_sentences(df['datasetName'].str.lower())
            print("Finished dataset names!")
            description_embs = model.embed_sentences(df['description'].str.lower())
            print("Finished descriptions!")
            all_embeddings = np.hstack((dataset_name_embs, descriptions_embs, col_name_embs))
            
            
    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=all_embeddings, columns=['emb_{}'.format(i) for i in range(len(all_embeddings))],index=index)

    return pd.concat([group_type_df, embeddings_df], axis=1)

if __name__=="__main__":
    embedding_model = 'sent2vec'
    #load the original csv
    _NUM_THREAD = (mp.cpu_count()-1)
    print("loading...")
    data_df = pd.read_csv(closed_d3m_file).applymap(str)
    #split the dataFrame into equal parts
    print("splitting...")
    splits = np.array_split(data_df, _NUM_THREAD)
    list_in = list()
    for i in range(len(splits)):
        list_in.append((splits[i], model_weights_path, embedding_model = embedding_model, use_col_name_only = True))
    print("starting multiprocess")
    mp_pool = mp.Pool()
    results = mp_pool.starmap(embed, list_in)
    mp_pool.close()
    mp_pool.join()
    
    df = pd.concat(results,axis=0)
    #save the results
    df.to_csv('closed_embed_'+embedding_model+'.csv',index=False)
    
