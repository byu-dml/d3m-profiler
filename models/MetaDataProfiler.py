import numpy as np
import pandas as pd
from models.ModelBase import ModelBase
from d3m_profiler.rebalance import rebalance_SMOTE as rebalance
from sentence_transformers import SentenceTransformer
import sent2vec


class MetaDataProfiler(ModelBase):
    def __init__(self, model, use_col_name_only: str, embedding_type: str, EMBEDDING_WEIGHTS_PATH: str, model_name: str, balance_type: str, balance: bool, embed_data_file: str):
        super().__init__()
        if (use_col_name_only is True):
            self.X_labels = ['colName']
        else:
            self.X_labels = ['colName', 'datasetName', 'description']
        self.EMBEDDING_WEIGHTS_PATH = EMBEDDING_WEIGHTS_PATH
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.model = model
        self.balance_type = balance_type
        self.balance = balance
        self.embed_data_file = embed_data_file
        self.loaded_embedding_model = False

    def fit(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        balanced = len(np.unique(counts)) == 1
        if (self.balance):
            if not balanced:
                X, y = rebalance(X, y, self.balance_type)
        self.model.fit(X, y)

    def encode_data(self, X):
        if (self.embedding_type == 'sent2vec'):
            if (self.loaded_embedding_model is False):
                self.embedding_model = sent2vec.Sent2vecModel()
                self.embedding_model.load_model(self.EMBEDDING_WEIGHTS_PATH)
            embedding = self.embedding_model.embed_sentences(X.str.lower())
        elif (self.embedding_type == 'SentenceTransformer'):
            if (self.loaded_embedding_model is False):
                self.embedding_model = SentenceTransformer(self.EMBEDDING_WEIGHTS_PATH)
            embedding = self.embedding_model.encode(X.str.lower())
        else:
            raise ValueError("{} is not a valid embedding_type".format(self.embedding_type))
        self.loaded_embedding_model = True
        return embedding

    def predict(self, X):
        return self.model.predict(X)
