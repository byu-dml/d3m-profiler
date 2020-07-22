from ModelBase import ModelBase
import os.path
import numpy as np
from multiprocessing import cpu_count

from Simon import Simon
from Simon.Encoder import Encoder

NUM_THREADS = cpu_count()


class BaselineSimon(ModelBase):
    def __init__(self, num_epochs=5, batch_size=64, max_cells=100, max_len=20):
        super().__init__()
        self.P_THRESHOLD = 0.3
        self.MAX_CELLS = max_cells
        self.MAX_LEN = max_len
        self.CHECKPOINT_DIR = './simon_pretrained_models/'
        if not os.path.isdir(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.encoder = None
        self.classifier = None
        self.model = None

    def encode_data(self, X, y):
        self.encoder = Encoder(categories=np.unique(y))
        self.encoder.process(X, self.MAX_CELLS)
        X, y = self.encoder.encode_data(X, y, self.MAX_LEN)
        return X, y

    def fit(self, X, y):
        encoder_max_cells = self.encoder.cur_max_cells
        category_count = y.shape[1]
        self.classifier = Simon(encoder=self.encoder)
        data = self._setup_test_sets(X, y)
        self.model = self.classifier.generate_model(self.MAX_LEN, encoder_max_cells, category_count)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        self.classifier.train_model(self.batch_size, self.CHECKPOINT_DIR, self.model, self.num_epochs, data)

    def predict(self, X):
        probabilities = self.model.predict(X, verbose=1)
        prediction_indices = probabilities > self.P_THRESHOLD
        y_pred = np.zeros(probabilities.shape)
        y_pred[prediction_indices] = 1
        return y_pred

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
