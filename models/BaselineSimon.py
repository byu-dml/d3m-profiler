from models.ModelBase import ModelBase
import os.path
import numpy as np
from Simon import Simon
from Simon.Encoder import Encoder
import time
import pandas as pd
import pickle
import warnings
from mpi4py import MPI
warnings.filterwarnings("ignore")

class BaselineSimon(ModelBase):
    def __init__(self, split_type, embed_data_file: str, data_path: str, num_epochs=5, batch_size=64, max_cells=100, max_len=20, model_name='SIMON', pkl=True):
        super().__init__()
        COMM = MPI.COMM_WORLD
        self.P_THRESHOLD = 0.3
        self.MAX_CELLS = max_cells
        self.MAX_LEN = max_len
        self.CHECKPOINT_DIR = './simon_{}/Simon/pretrained_models/'.format(COMM.rank)
        if not os.path.isdir(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)
        self.to_drop = ['colType']
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.split_type = split_type
        self.encoder = None
        self.classifier = None
        self.model_simon = None
        self.model_name = model_name
        self.pkl = pkl
        self.embed_data_file = embed_data_file
        self.data_path = data_path
        self.map_path = 'simon_map.pkl'

    def encode_data(self, X, y):
        start = time.time()
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
        simon_y = pd.DataFrame(columns=['colType'])
        simon_y['colType'] = list(data_simon_y)
        end = time.time()
        print("Time to encode: {:.2f}".format(end-start))
        return simon_data_X, simon_y

    def fit(self, X, y):
        print("Fitting Model: {}".format(model_name))
        y_simon = np.vstack(y['colType'].tolist())
        X_simon = np.asarray(list(X['embedding']))
        encoder_max_cells = X_simon.shape[1]
        category_count = y_simon.shape[1]
        self.classifier = Simon(encoder=self.encoder)
        data = self._setup_test_sets(X_simon, y_simon)
        self.model_simon = self.classifier.generate_model(self.MAX_LEN, encoder_max_cells, category_count)
        self.model_simon.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        self.classifier.train_model(self.batch_size, self.CHECKPOINT_DIR, self.model_simon, self.num_epochs, data)

    def predict(self, X):
        probabilities = self.model_simon.predict(np.asarray(list(X['embedding'])))
        y_pred = np.zeros(probabilities.shape, dtype=int)
        y_pred[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
        mapping = pickle.load(open(self.map_path,"rb"))
        y_pred_final = self._map_results(np.argmax(y_pred, axis=1), mapping)
        print(y_pred_final)
        return y_pred_final
        
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
