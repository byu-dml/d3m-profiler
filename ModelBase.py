import abc


class ModelBase:
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def encode_data(self, X, y):
        pass