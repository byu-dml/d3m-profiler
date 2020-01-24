import logging, csv

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

from build_table import build_table
from embed import embed

logger = logging.getLogger(__name__)

def run_model(model_class, x_train, x_test, y_train, y_test):
    model = model_class()
    # print('Fitting {}'.format(model_class))
    model.fit(x_train, y_train)
    # print('Done fitting model')
    # print('Predicting colTypes')
    preds = model.predict(x_test)

    print(accuracy_score(y_test, preds))

    disp = plot_confusion_matrix(model, x_test, y_test, display_labels=y_test.unique() ,normalize='true')

    print(y_test.unique())

    print(disp.confusion_matrix)
    plt.show()
    plt.close()



if __name__ == '__main__':
    print('Built table complete')
    print('Embedding complete')
    # data = pd.read_csv('train_data_embedded.csv')
    data = pd.read_csv('/users/data/d3m/embedded_data_small.csv')

    target=data.colType
    features=data.drop('colType', axis=1)

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

    run_model(svm.SVC, features_train, features_test, target_train, target_test)
    run_model(RandomForestClassifier, features_train, features_test, target_train, target_test)
