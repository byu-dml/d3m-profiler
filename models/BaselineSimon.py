from models.ModelBase import ModelBase
import os.path
import numpy as np
from Simon import Simon
from Simon.Encoder import Encoder
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

class BaselineSimon(ModelBase):
    def __init__(self, split_type, embed_data_file: str, data_path: str, num_epochs=5, batch_size=64, max_cells=100, max_len=20, model_name='SIMON', pkl=True):
        super().__init__()
        self.P_THRESHOLD = 0.3
        self.MAX_CELLS = max_cells
        self.MAX_LEN = max_len
        self.CHECKPOINT_DIR = './simon/Simon/pretrained_models/'
        if not os.path.isdir(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.split_type = split_type
        self.encoder = None
        self.classifier = None
        self.model = None
        self.model_name = model_name
        self.pkl = pkl
        self.embed_data_file = embed_data_file
        self.data_path = data_path
        self.map_path = 'simon_map.pkl'
        self.map = True

    def encode_data(self, raw_data, header):
        raw_data = np.asarray(list(raw_data['values']))
        header = pd.DataFrame(header).to_numpy()
        unique_list = np.unique(header)
        dict_map = {it:unique_list[it] for it in range(len(unique_list))}
        pickle.dump(dict_map, open(self.map_path, 'wb'))
        self.encoder = Encoder(categories=unique_list)
        self.encoder.process(raw_data, self.MAX_CELLS)
        X, y = self.encoder.encode_data(raw_data, header, self.MAX_LEN)
        data_x = pd.DataFrame(columns=['embedding'])
        data_x['embedding'] = list(X)
        data_y = pd.DataFrame(columns=['colType'])
        data_y['colType'] = list(y)
        return data_x, data_y

    def fit(self, X, y):
        y = np.vstack(y.tolist())
        X = np.vstack(X.tolist()) 
        encoder_max_cells = X.shape[1]
        category_count = y.shape[1]
        self.classifier = Simon(encoder=self.encoder)
        data = self._setup_test_sets(X, y)
        self.model = self.classifier.generate_model(self.MAX_LEN, encoder_max_cells, category_count)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        self.classifier.train_model(self.batch_size, self.CHECKPOINT_DIR, self.model, self.num_epochs, data)

    def predict(self, X):
        X = np.vstack(X.tolist())
        probabilities = self.model.predict(X, verbose=1)
        prediction_indices = probabilities > self.P_THRESHOLD
        y_pred = np.zeros(probabilities.shape, dtype=int)
        y_pred[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
        mapping = pickle.load(open(self.map_path,"rb"))
        y_pred_final = self._map_results(np.argmax(y_pred, axis=1), mapping)
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
