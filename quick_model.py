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

# load data
data = pd.read_csv(data_path)#.iloc[:1000]

# drop useless columns
data.drop(['description'], axis=1, inplace=True)
print(data.shape)

def sep_X_and_y(data):
    X = data[['colName']]
    y = data['colType']
    return X, y

scores = []

# split data
splitter = GroupShuffleSplit(n_splits=2, train_size=0.66, random_state=42)
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
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42) # todo, n_jobs=_NUM_THREADS)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_emb, y_train)
    print(len(X_train_bal))

    # create pca/rf pipeline
    pca = PCA(n_components=50)
    rf = RandomForestClassifier(max_depth=10)
    pipeline = Pipeline(steps=[('pca', pca), ('rf', rf)])

    # train pipeline
    pipeline.fit(X_train_bal, y_train_bal)
    y_test_hat = pipeline.predict(X_test_emb)

    # score pipeline
    conf_mat = confusion_matrix(y_test, y_test_hat, labels=y_train.unique())
    print(conf_mat)
    score = f1_score(y_test, y_test_hat, labels=y_train.unique(), average='macro')
    print(f'Fold {i} F1 macro: {score}')
    scores.append(score)

print(f'Mean F1 Score: {np.mean(scores)}')

# save pipeline
with open(model_save_path, 'wb') as f:
    pickle.dump(pipeline, f)
