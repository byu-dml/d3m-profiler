from models.ModelBase import ModelBase
import os.path
import numpy as np
from Simon import Simon
from Simon.Encoder import Encoder
import pandas as pd
import pickle
import warnings
import time
from mpi4py import MPI
warnings.filterwarnings("ignore")
from d3m_profiler.rebalance import rebalance_SMOTE as rebalance
import tensorflow as tf
from sentence_transformers import SentenceTransformer


class MetaSimon(ModelBase):
    def __init__(self, data_path, embed_data_path, model_meta, model_both, balance, split_type, seed):
        super().__init__()
        self.data_path = data_path
        self.embed_data_file = embed_data_path
        self.model_meta = model_meta
        self.model_both = model_both
        self.model_simon = None
        self.embedding_weights_path = '../data_files/SentenceTransformer'
        self.balance = balance
        self.model_name = 'simon_rf_mlp_model'
        self.balance_type = 'SMOTE'
        self.map_path = 'simon_map.pkl'
        self.pkl = True
        self.to_drop = ['colType','colTypeHot']
        COMM = MPI.COMM_WORLD
        self.P_THRESHOLD = 0.3
        self.MAX_CELLS = 100
        self.MAX_LEN = 20
        self.CHECKPOINT_DIR = './simon/Simon_{}/pretrained_models/'.format(COMM.rank)
        if not os.path.isdir(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)

        self.num_epochs = 5
        self.batch_size = 64
        self.split_type = split_type
        self.encoder = None
        self.classifier = None
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)
        
    def encode_data(self, X, y):
        start = time.time()
        #do the metadata encoding first
        self.embedding_model = SentenceTransformer(self.embedding_weights_path)
        embedding_meta = self.embedding_model.encode(X['colName'].apply(str).str.lower())
        embedding_meta = pd.DataFrame(data=embedding_meta, columns=['emb_{}'.format(i) for i in range(len(embedding_meta[0]))])
        #do the simon encoding next!
        simon_data = np.asarray(list(X['values']))
        header = pd.DataFrame(y).to_numpy()
        unique_list = np.unique(header)
        dict_map = {it:unique_list[it] for it in range(len(unique_list))}
        pickle.dump(dict_map, open(self.map_path, 'wb'))
        self.encoder = Encoder(categories=unique_list)
        self.encoder.process(simon_data, self.MAX_CELLS)
        simon_data_X, data_simon_y = self.encoder.encode_data(simon_data, header, self.MAX_LEN)
        simon_data = pd.DataFrame(columns=['embedding'])
        simon_data['embedding'] = list(simon_data_X)
        simon_y = pd.DataFrame(columns=['colTypeHot'])
        simon_y['colTypeHot'] = list(data_simon_y)
        end = time.time()
        #now concatenate the y and X encoding together
        y = pd.concat([simon_y, y],axis=1)
        X = pd.concat([simon_data, embedding_meta],axis=1)
        print("Time to embed: {:.2f}".format(end-start))
        return X,y
        
    def fit(self, X, y):
        print("Fitting Model")
        #fit metadata first
        unique, counts = np.unique(y['colType'], return_counts=True)
        balanced = len(np.unique(counts)) == 1
        if (self.balance):
            if not balanced:
                X_meta, y_meta = rebalance(X.drop(['embedding'],axis=1).to_numpy(), y['colType'], self.balance_type)
                y_meta = np.asarray(y_meta)
                print(y_meta)
        else:
            X_meta = X.drop(['embedding'],axis=1).to_numpy()
            y_meta = y['colType'].to_numpy()
        self.model_meta.fit(X_meta, y_meta)
        #now fit simon
        y_simon = np.vstack(y['colTypeHot'].tolist())
        X_simon = np.asarray(list(X['embedding']))
        encoder_max_cells = X_simon.shape[1]
        category_count = y_simon.shape[1]
        self.classifier = Simon(encoder=self.encoder)
        data = self._setup_test_sets(X_simon, y_simon, random_state=self.seed)
        self.model_simon = self.classifier.generate_model(self.MAX_LEN, encoder_max_cells, category_count)
        self.model_simon.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        self.classifier.train_model(self.batch_size, self.CHECKPOINT_DIR, self.model_simon, self.num_epochs, data)
        #now get the data from both
        mapping = pickle.load(open(self.map_path,"rb"))
        inv_map = {v : k for k, v in mapping.items()}
        y_hat_meta = self._one_hot(np.asarray([inv_map[i] for i in self.model_meta.predict(X.drop(['embedding'],axis=1).to_numpy())]))
        probabilities = self.model_simon.predict(np.asarray(list(X['embedding'])))
        self.model_both.fit(np.hstack([probabilities, y_hat_meta]), y['colType'])
        print("Done Fitting Model")
        
    def predict(self, X):
        #now get the data from both
        mapping = pickle.load(open(self.map_path,"rb"))
        inv_map = {v : k for k, v in mapping.items()}
        y_hat_meta = self._one_hot(np.asarray([inv_map[i] for i in self.model_meta.predict(X.drop(['embedding'],axis=1).to_numpy())]))
        probabilities = self.model_simon.predict(np.asarray(list(X['embedding'])))
        y_hat = self.model_both.predict(np.hstack([probabilities, y_hat_meta]))
        print(y_hat)
        return y_hat
        
    @staticmethod
    def _setup_test_sets(X, y, random_state=None):
        ids = np.arange(len(X))
        if random_state:
            np.random.seed(random_state)
        np.random.shuffle(ids)

        # shuffle
        X = X[ids]
        y = y[ids]

        train_end = int(X.shape[0] * .7)
        cross_validation_end = int(X.shape[0] * .3 + train_end)

        X_train = X[:train_end]
        X_cv_test = X[train_end:cross_validation_end]

        y_train = y[:train_end]
        y_cv_test = y[train_end:cross_validation_end]
        data = type('data_type', (object,),
                    {'X_train': X_train, 'X_cv_test': X_cv_test, 'y_train': y_train, 'y_cv_test': y_cv_test})
        return data
        
    @staticmethod
    def _map_results(y, mapping):
        y_results = [mapping[i] for i in y]
        return y_results
        
    @staticmethod
    def _one_hot(y):
        b = np.zeros((y.size, y.max()+1))
        b[np.arange(y.size),y]=1
        return b
        
        
        
