import numpy as np
import pandas as pd
from models.ModelBase import ModelBase
from d3m_profiler.rebalance import rebalance_SMOTE as rebalance
from sentence_transformers import SentenceTransformer
import sent2vec
import warnings
warnings.filterwarnings("ignore")


class MetaDataProfiler(ModelBase):
    def __init__(self, model, use_col_name_only: str, embedding_type: str, EMBEDDING_WEIGHTS_PATH: str, model_name: str, balance_type: str, balance: bool, embed_data_file: str, split_type, data_path: str, pkl=False):
        super().__init__()
        if (use_col_name_only is True):
            self.X_labels = ['colName']
        else:
            self.X_labels = ['colName', 'datasetName', 'description']
        self.to_drop = ['colType']
        self.EMBEDDING_WEIGHTS_PATH = EMBEDDING_WEIGHTS_PATH
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.model = model
        self.balance_type = balance_type
        self.balance = balance
        self.embed_data_file = embed_data_file
        self.loaded_embedding_model = False
        self.split_type = split_type
        self.pkl=pkl
        self.data_path = data_path

    def fit(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        balanced = len(np.unique(counts)) == 1
        if (self.balance):
            if not balanced:
                X, y = rebalance(X, y['colType'], self.balance_type)
        else:
            X = X.to_numpy()
            y = y['colType'].to_numpy()
        self.model.fit(X, y)

    def encode_data(self, X, y):
        X = X[self.X_labels]
        if (self.embedding_type == 'sent2vec'):
            self.embedding_model = sent2vec.Sent2vecModel()
            self.embedding_model.load_model(self.EMBEDDING_WEIGHTS_PATH)
            embeddings = []
            for i in X.columns:
                embedding = self.embedding_model.embed_sentences(X[i].apply(str).str.lower())
                embeddings.append(embedding)
            X_embed = pd.DataFrame(data=np.hstack(tuple(embeddings)), columns=['emb_{}'.format(i) for i in range(len(embeddings)*len(embeddings[0][0]))])
        elif (self.embedding_type == 'SentenceTransformer'):
            self.embedding_model = SentenceTransformer(self.EMBEDDING_WEIGHTS_PATH)
            embeddings = []
            for i in X.columns:
                embedding = self.embedding_model.encode(X[i].apply(str).str.lower())
                embeddings.append(embedding)
            X_embed = pd.DataFrame(data=np.hstack(tuple(embeddings)), columns=['emb_{}'.format(i) for i in range(len(embeddings)*len(embeddings[0][0]))])
        else:
            raise ValueError("{} is not a valid embedding_type".format(self.embedding_type))
        return X_embed, y

    def predict(self, X):
        return self.model.predict(X)
