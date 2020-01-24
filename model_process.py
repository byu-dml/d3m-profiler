from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

from embed import EMBEDDED_DATA_PATH, EMBEDDED_SMALL_DATA_PATH

def load_data(small=True):
    if small:
        data_path = EMBEDDED_SMALL_DATA_PATH
    else:
        data_path = EMBEDDED_DATA_PATH

    with open(data_path, 'r') as f:
        data = pd.read_csv(f)

    dataset_names = data['datasetName']
    y = data['colType']
    X = data.drop(['datasetName', 'colType'], axis=1)

    return dataset_names, X, y


def evaluate_model(model_constructor, dataset_names, X, y):
    n_splits = np.unique(dataset_names).shape[0]
    group_kfold = GroupKFold(n_splits=3)  # TODO: use leave on out CV

    cv_train_accuracies = []
    cv_test_accuracies = []
    cv_train_confusion_matrices = []
    cv_test_confusion_matrices = []
    for i, (train_indices, test_indices) in enumerate(group_kfold.split(X, y, dataset_names)):  # TODO: parallelize
        print('fold {}'.format(i))
        X_train, y_train = X.values[train_indices], y.values[train_indices]
        X_test, y_test = X.values[test_indices], y.values[test_indices]

        model = model_constructor()
        model.fit(X_train, y_train)

        # TODO: figure out confusion matrices, make sure 1) we know the labels of each row/col 2) the matrices are aligned between folds

        y_train_pred = model.predict(X_train)
        cv_train_accuracies.append(accuracy_score(y_train, y_train_pred))
        # cv_train_confusion_matrices.append(confusion_matrix(y_train, y_train_pred, normalize='true', labels=np.unique(y)))

        y_test_pred = model.predict(X_test)
        cv_test_accuracies.append(accuracy_score(y_test, y_test_pred))
        # cv_test_confusion_matrices.append(confusion_matrix(y_test, y_test_pred, normalize='true', labels=np.unique(y)))

    return np.mean(cv_train_accuracies), np.mean(cv_test_accuracies)#, np.mean(cv_train_confusion_matrices, axis=0), np.mean(cv_test_confusion_matrices, axis=0)


if __name__ == '__main__':
    print('loading data...')
    dataset_names, X, y = load_data(False)

    for model_class in [svm.SVC, RandomForestClassifier]:
        print('evaluating model {}'.format(model_class))
        avg_train_acc, avg_test_acc = evaluate_model(model_class, dataset_names, X, y)
        print('train accuracy: {}'.format(avg_train_acc))
        # print(avg_train_conf_matrix)
        print('test accuracy: {}'.format(avg_test_acc))
        # print(avg_test_conf_matrix)
