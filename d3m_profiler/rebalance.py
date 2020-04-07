import openml
import csv
import pandas as pd
from sklearn.utils import random
import sklearn
import random
from datetime import datetime
from typing import Callable

def get_openml100_data():
    openml100_suite = openml.study.get_suite('OpenML100')
    dataset = openml.datasets.get_dataset(openml100_suite.data[0])
    name = dataset.name
    description = dataset.description
    features = dataset.features
    
    
    with open('test_data.csv', 'w+', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerow(['datasetName', 'description', 'colName', 'colType'])
        for feature in features.values():
            feature_name = feature.name
            feature_type = feature.data_type
            datawriter.writerow([name, description, feature_name, feature_type])
            
def undersample(df: pd.DataFrame, type_column: str, majority_class: str, n_samples: int) -> pd.DataFrame:
    majority_indices = df.groupby(type_column).indices[majority_class]
    n_majority_indices = len(majority_indices)
    
    n_to_drop = n_majority_indices - n_samples
    
    indicies_to_drop = sklearn.utils.resample(majority_indices, n_samples=n_to_drop, replace=False)
    
    return df.drop(indicies_to_drop)

def oversample(df: pd.DataFrame, type_column: str, minority_class: str, n_samples: int) -> pd.DataFrame:
    minority_indices = df.groupby(type_column).indices[minority_class]
    df_minority = df.iloc[minority_indices]
    
    resampled_df_minority = sklearn.utils.resample(df_minority, n_samples=n_samples)
    
    return df.append(resampled_df_minority)

def configure_reablance_callables(method: str, class_counts: pd.Series):
    if (method == "under"):
        return class_counts.idxmin, class_counts.min, (lambda x,y: x), undersample
    elif (method == "over"):
        return class_counts.idxmax, class_counts.max, (lambda x,y: x - y), oversample
    else:
        raise ValueError('\"{}\" is invalid argument for method parameter\n\tValid arguments: [\"under\", \"over\"]'.format(method))
    
def rebalance(df: pd.DataFrame, type_column: str, method: str) -> pd.DataFrame:
    class_counts = df[type_column].value_counts()
    idx, idx_count, calc_samples, rebalance = configure_reablance_callables(method, class_counts)

    classes = df[type_column].unique()        
    critical_class = idx()
    critical_class_count = idx_count()
    
    non_critical_classes = [x for x in classes if x != critical_class]
    
    for non_critical_class in non_critical_classes:
        non_critical_class_count = class_counts[non_critical_class]
        n_samples = calc_samples(critical_class_count, non_critical_class_count)
        df = rebalance(df, type_column, non_critical_class, n_samples)
    return df
    
if __name__ == '__main__':
    df = pd.read_csv('test_data.csv')
    df = df.append({'datasetName': 'TEST_DATASETNAME', 'description': 'TEST_DESCRIPTION', 'colName': 'TEST_COLNAME', 'colType': 'integer'}, ignore_index=True)
    
    df = rebalance(df, 'colType', 'over')
    
    print(df)
    
