import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from ModelBase import ModelBase
from d3m_profiler.embed import embed
from d3m_profiler.rebalance import rebalance_SMOTE as rebalance


class MetadataProfiler(ModelBase):
    def __init__(self, model=RandomForestClassifier):
        super().__init__()
        self.model_weights_path = os.getenv('MODEL_WEIGHTS_DIR', '') + 'torontobooks_unigrams.bin'
        self.model = model()
        self.rebalance_type = None
        self.X_metadata_labels = None

    def fit(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        balanced = len(np.unique(counts)) == 1
        if not balanced:
            try:
                X, y = rebalance(X, y, 'smote')
                self.rebalance_type = 'smote'
            except ValueError as e:
                # TODO run some other kind of rebalancing
                self.rebalance_type = 'none'
                print(e)
        self.model.fit(X, y)

    def encode_data(self, X, y):
        self.X_metadata_labels = list(X.columns)
        return embed(X, self.model_weights_path), y

    def predict(self, X):
        return self.model.predict(X)

    def get_note(self):
        return f'rebalance: {self.rebalance_type}; columns: {self.X_metadata_labels}'
