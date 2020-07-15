from d3m_profiler import evaluate_models
from models.model_types import get_model 

#the file that contains the unembedded data
file_data = '../data_files/data/closed_d3m_data.csv'

#create the models
#model_1=get_model(n_splits=None, embed_type='SentenceTransformer', use_col=True, split_name='LeaveOneGroupOut', model_type='RandomForestClassifier', balance=True)
model_2=get_model(n_splits=10, embed_type='SentenceTransformer', use_col=True, split_name='GroupShuffleSplit', model_type='RandomForestClassifier', balance=True)
model_3=get_model(n_splits=2, embed_type='SentenceTransformer', use_col=True, split_name='ShuffleSplit', model_type='RandomForestClassifier', balance=True)
#model_4=get_model(n_splits=None, embed_type='SentenceTransformer', use_col=False, split_name='LeaveOneGroupOut', model_type='RandomForestClassifier', balance=True)
model_5=get_model(n_splits=10, embed_type='SentenceTransformer', use_col=False, split_name='GroupShuffleSplit', model_type='RandomForestClassifier', balance=True)
model_6=get_model(n_splits=2, embed_type='SentenceTransformer', use_col=False, split_name='ShuffleSplit', model_type='RandomForestClassifier', balance=True)
#model_7=get_model(n_splits=None, embed_type='sent2vec', use_col=True, split_name='LeaveOneGroupOut', model_type='RandomForestClassifier', balance=True)
model_8=get_model(n_splits=10, embed_type='sent2vec', use_col=True, split_name='GroupShuffleSplit', model_type='RandomForestClassifier', balance=True)
model_9=get_model(n_splits=2, embed_type='sent2vec', use_col=True, split_name='ShuffleSplit', model_type='RandomForestClassifier', balance=True)
#model_10=get_model(n_splits=None, embed_type='sent2vec', use_col=False, split_name='LeaveOneGroupOut', model_type='RandomForestClassifier', balance=True)
model_11=get_model(n_splits=10, embed_type='sent2vec', use_col=False, split_name='GroupShuffleSplit', model_type='RandomForestClassifier', balance=True)
model_12=get_model(n_splits=2, embed_type='sent2vec', use_col=False, split_name='ShuffleSplit', model_type='RandomForestClassifier', balance=True)

#get both the models to run with the profiler
initialized_models = [model_2,model_3,model_5,model_6,model_8,model_9,model_11,model_12]
#calling this will save all of the initialized model score results to a csv called 'models_final_cross_val.csv'
evaluate_models.run_models(initialized_models=initialized_models, data_path=file_data)
