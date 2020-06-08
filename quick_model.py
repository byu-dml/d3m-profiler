import pickle

import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from d3m_profiler.embed_data import embed

# config
data_path = 'closed_d3m_data.csv'
embed_model_path = './distilbert-base-nli-stsb-mean-tokens'
k_neighbors = 3
model_save_path = './quick_model.bin'

# load data
data = pd.read_csv(data_path)#.iloc[:1000]

# drop useless columns
data.drop(['description'], axis=1, inplace=True)
print(data.shape)

# split data
group_k_fold = GroupKFold(n_splits=5)
train_indices, test_indices = next(group_k_fold.split(data, groups=data['datasetName']))
train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]

# separate x and y
def sep_X_and_y(data):
    X = data[['colName']]
    y = data['colType']
    return X, y

X_train, y_train = sep_X_and_y(train_data)
X_test, y_test = sep_X_and_y(test_data)

# lower case col names
X_train['colName'] = X_train['colName'].str.lower()
X_test['colName'] = X_test['colName'].str.lower()

print(X_train)
print(y_train)

# embed data
embed_model = SentenceTransformer(embed_model_path)
X_train_emb = embed_model.encode(X_train['colName'].to_numpy())
X_test_emb = embed_model.encode(X_test['colName'].to_numpy())

# balance train data
smote = SMOTE(k_neighbors=k_neighbors) # todo, n_jobs=_NUM_THREADS) random_state=42,
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
print(f1_score(y_test, y_test_hat, labels=y_train.unique(), average='macro'))

# save pipeline
with open(model_save_path, 'wb') as f:
    pickle.dump(pipeline, f)
