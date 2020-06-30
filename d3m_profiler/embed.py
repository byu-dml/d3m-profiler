import sys
from sentence_transformers import SentenceTransformer
import sent2vec
import numpy as np
import pandas as pd

"""
Initializes a model based on model weights
"""
def embed(df: pd.DataFrame, model_weights_path: str, embedding_model: str, use_col_name_only: bool):
    index = df.index.to_list()
    dataset_names = df['datasetName']
    col_types = df['colType']
    print("Starting Embedding")
    if (embedding_model == 'SentenceTransformer'):
        model = SentenceTransformer(model_weights_path)
        col_name_embs = model.encode(df['colName'].str.lower().to_numpy())
        all_embeddings = np.array(col_name_embs)
        if (use_col_name_only is False):
            dataset_name_embs = model.encode(df['datasetName'].str.lower().to_numpy())
            print("Finished dataset names!")
            description_embs = model.encode(df['description'].str.lower().to_numpy())
            print("Finished descriptions!")
            all_embeddings = np.hstack((dataset_name_embs, description_embs, col_name_embs))
        
    elif (embedding_model == 'sent2vec'):
        model = sent2vec.Sent2vecModel()
        model.load_model(model_weights_path)
        col_name_embs = model.embed_sentences(df['colName'].str.lower())
        all_embeddings = np.array(col_name_embs)
        if (use_col_name_only is False):
            dataset_name_embs = model.embed_sentences(df['datasetName'].str.lower())
            print("Finished dataset names!")
            description_embs = model.embed_sentences(df['description'].str.lower())
            print("Finished descriptions!")
            all_embeddings = np.hstack((dataset_name_embs, description_embs, col_name_embs))
            
            
    group_type_df = pd.DataFrame({'datasetName': dataset_names, 'colType': col_types})
    embeddings_df = pd.DataFrame(data=all_embeddings, columns=['emb_{}'.format(i) for i in range(len(all_embeddings[0]))],index=index)
    return pd.concat([group_type_df, embeddings_df], axis=1)

