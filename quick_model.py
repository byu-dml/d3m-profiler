import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

# config
data_path = 'closed_d3m_data.csv'
embed_model_path = 'distilbert-base-nli-stsb-mean-tokens'
default_k_neighbors = 3
model_save_path = './quick_model.bin'
random_state=42

# load data
data = pd.read_csv(data_path)#.iloc[:1000]

# drop useless columns
data.drop(['description'], axis=1, inplace=True)
print(data.shape)

def sep_X_and_y(data):
    X = data[['colName']]
    y = data['colType']
    return X, y

f1_macros = []
f1_micros = []
f1_weighted = []
conf_matrices = []
conf_matrices_norm = []

# split data
splitter = GroupShuffleSplit(n_splits=9, train_size=0.66, random_state=random_state)
for i, (train_indices, test_indices) in enumerate(splitter.split(data, groups=data['datasetName'])):
    print(f'Fold {i}')
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    # separate x and y
    X_train, y_train = sep_X_and_y(train_data)
    X_test, y_test = sep_X_and_y(test_data)

    # lower case col names
    X_train['colName'] = X_train['colName'].str.lower()
    X_test['colName'] = X_test['colName'].str.lower()

    print(X_train)
    print(y_train)
    k_neighbors = min(min(y_train.value_counts()) - 1, default_k_neighbors)

    # embed data
    embed_model = SentenceTransformer(embed_model_path)
    X_train_emb = embed_model.encode(X_train['colName'].to_numpy())
    X_test_emb = embed_model.encode(X_test['colName'].to_numpy())

    # balance train data
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state) # todo, n_jobs=_NUM_THREADS)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_emb, y_train)
    print(len(X_train_bal))

    # create pca/rf pipeline
    pca = PCA(n_components=50, random_state=random_state)
    rf = RandomForestClassifier(max_depth=10, random_state=random_state)
    pipeline = Pipeline(steps=[('pca', pca), ('rf', rf)])

    # train pipeline
    pipeline.fit(X_train_bal, y_train_bal)
    y_test_hat = pipeline.predict(X_test_emb)

    # score pipeline
    conf_mat = confusion_matrix(y_test, y_test_hat, labels=y_train.unique())
    print(conf_mat)
    f1_macros.append(f1_score(y_test, y_test_hat, labels=y_train.unique(), average='macro'))
    f1_micros.append(f1_score(y_test, y_test_hat, labels=y_train.unique(), average='micro'))
    f1_weighted.append(f1_score(y_test, y_test_hat, labels=y_train.unique(), average='weighted'))
    conf_matrices.append(conf_mat)
    conf_matrices_norm.append(confusion_matrix(y_test, y_test_hat, labels=y_train.unique(), normalize='all'))

print(f'F1 Macros: {f1_macros}')
print(f'F1 Micros: {f1_micros}')
print(f'F1 Weighteds: {f1_weighted}')
np.set_printoptions(precision=5, suppress=True)
print(f'Average confusion matrix: {np.mean(conf_matrices, axis=0)}')
print(f'Average normalized confusion matrix: {np.mean(conf_matrices_norm, axis=0)}')
print(f'Mean F1 Macro: {np.mean(f1_macros)}')
print(f'Mean F1 Micro: {np.mean(f1_micros)}')
print(f'Mean F1 Weighted: {np.mean(f1_weighted)}')

# save pipeline
with open(model_save_path, 'wb') as f:
    pickle.dump(pipeline, f)
