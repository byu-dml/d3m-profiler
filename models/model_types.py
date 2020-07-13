from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ShuffleSplit
from models.MetaDataProfiler import MetaDataProfiler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#create the models
def get_model(n_splits=None, embed_type='SentenceTransformer', use_col=True, split_name='LeaveOneGroupOut', model_type='RandomForestClassifier', balance=True):
    #get the splitter
    if (split_name == 'LeaveOneGroupOut'):
        split_type = LeaveOneGroupOut()
    elif (split_name == 'GroupShuffleSplit'):
        if (n_splits is None):
            raise ValueError("Must include n_folds parameter for {}".format(split_name))
        split_type = GroupShuffleSplit(n_splits=n_splits, train_size=0.7, random_state=15)
    elif (split_name == 'ShuffleSplit'):
        if (n_splits is None):
            raise ValueError("Must include n_folds parameter for {}".format(split_name))
        split_type = ShuffleSplit(n_splits=n_splits, train_size=0.7, random_state=14)
    else:
        raise ValueError("Not a valid shuffle type")
    #get the model    
    if (model_type == 'RandomForestClassifier'):
        model=RandomForestClassifier(random_state=5)
    elif (model_type == 'MLPClassifier'):
        model=MLPClassifier(random_state=21)
    else:
        raise ValueError("Not a valid classification model type")
        
    model = MetaDataProfiler(use_col_name_only=use_col, embedding_type=embed_type, EMBEDDING_WEIGHTS_PATH='../data_files/{}'.format(embed_type), model=model, model_name='{}/{}/{}_{}/col_{}'.format(model_type, embed_type, split_name, n_splits, use_col), balance_type='SMOTE', balance=balance, embed_data_file='{}_embedding_col_{}.csv'.format(embed_type, use_col), split_type=split_type)
    
    return model
