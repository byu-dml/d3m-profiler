import multiprocessing as mp
import pickle as pk
import sys
#import h5py
#pip install absl-py
#pip install tensorflow
#pip install tensorflow_hub
#pip install tqdm
import tensorflow as tf
import tensorflow_hub as hub
#from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm

_NUM_THREAD = (mp.cpu_count() - 1)

tf.compat.v1.disable_eager_execution()
model_weights_path = '../../data_files/elmo_3'
closed_d3m_file = '../../data_files/data/closed_d3m_data.csv'
file = '../data_files/data/sample.csv'
open_d3m_file = '../../data_files/data/open_d3m_data.csv'

def initialize_model(model_path):
    model = hub.Module(model_path)
    return model
    
def embed(data, model_weights_path):
    model = initialize_model(model_weights_path)
    index = data.index.to_list()
    dataset_names = data['datasetName']
    print("starting embedding")
    #print(np.shape(data['colName'].str.lower().to_numpy()))
    col_name_embs = model(data['colName'].str.lower().to_numpy())
    #print(col_name_embs)
    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        col_embs = session.run(col_name_embs)
    col_types = data['colType']
    one, two = col_name_embs.get_shape()
    #print(one,two)
    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=col_embs, columns=['emb_{}'.format(i) for i in range(two)],index=index)
    del col_embs
    del col_name_embs
    return pd.concat([group_type_df, embeddings_df],axis=1)

if __name__=="__main__":
    #load the original csv
    print("loading...")
    data_df = pd.read_csv(closed_d3m_file).applymap(str)
    #data_df = data_df.sample(100)
    #split the dataFrame into equal parts
    print("splitting...")
    splits = np.array_split(data_df, _NUM_THREAD*50)
    list_in = list()
    for i in range(len(splits)):
        list_in.append((splits[i],model_weights_path))
    del splits
    del data_df
    print("starting multiprocess")
    mp_pool = mp.Pool()
    results = mp_pool.starmap(embed, list_in)
    del list_in
    mp_pool.close()
    mp_pool.join()
    df = pd.concat(results,axis=0)
    #save the results
    df.to_csv('closed_embed_elmo.csv',index=False)
