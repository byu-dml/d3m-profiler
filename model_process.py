import logging, csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


from build_table import build_table
from embed import embed

logger = logging.getLogger(__name__)

def svm_process(x_train, x_test, y_train, y_test):
    svm_clf = svm.SVC()
    print('Fitting SVM')
    svm_clf.fit(x_train, y_train)
    print('Done fitting SVM')
    print('Predicting colTypes')
    preds = svm_clf.predict(x_test)

    print(accuracy_score(y_test, preds))


if __name__ == '__main__':
    print('Built table complete')
    print('Embedding complete')
    data = pd.read_csv('train_data_embedded.csv')

    target=data.colType
    features=data.drop('colType', axis=1)

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

    svm_process(features_train, features_test, target_train, target_test)


def rf_processs(x_train, x_test, y_train, y_test):
    rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
