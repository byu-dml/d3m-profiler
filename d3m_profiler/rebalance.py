import multiprocessing as mp

import numpy as np
import pandas as pd
import sent2vec

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE

from d3m_profiler.embed import embed

_NUM_THREADS = (mp.cpu_count() - 1)

"""
Under samples majority class(es) to align with shape of minority class(es)

Returns
-------
df: pandas.DataFrame
    An undersampled DataFrame
"""            
def _undersample(df: pd.DataFrame, type_column: str, majority_class: str, n_samples: int) -> pd.DataFrame:
    majority_indices = df.groupby(type_column).indices[majority_class]
    n_majority_indices = len(majority_indices)
    
    n_to_drop = n_majority_indices - n_samples
    
    indicies_to_drop = sklearn.utils.resample(majority_indices, n_samples=n_to_drop, replace=False)
    
    return df.drop(indicies_to_drop)

"""
Over samples minority class(es) to align with shape of majority class(es)

Returns
-------
df: pandas.DataFrame
    An oversampled DataFrame
""" 
def _oversample(df: pd.DataFrame, type_column: str, minority_class: str, n_samples: int) -> pd.DataFrame:
    minority_indices = df.groupby(type_column).indices[minority_class]
    df_minority = df.iloc[minority_indices]
    
    resampled_df_minority = sklearn.utils.resample(df_minority, n_samples=n_samples)
    
    return df.append(resampled_df_minority)

"""
Configures callables based on the method of class rebalancing.
Valid arguments for "method" parameter are: ["under", "over"]

Returns
-------
tuple(idx, idx_count, calc_samples, rebalance_method): Tuple(Callables)
    The callables based on the method of class rebalancing
"""
def _configure_reablance_callables(method: str, class_counts: pd.Series):
    if (method == "under"):
        return class_counts.idxmin, class_counts.min, (lambda x,y: x), undersample
    elif (method == "over"):
        return class_counts.idxmax, class_counts.max, (lambda x,y: x - y), oversample
    else:
        raise ValueError('\"{}\" is invalid argument for method parameter\n\tValid arguments: [\"under\", \"over\"]'.format(method))
    
"""
Rebalances a DataFrame based on a certain method
Valid arguments for "method" parameter are: ["under", "over"]

Returns
-------
df: pandas.DataFrame
    The rebalanced DataFrame
"""
def rebalance(df: pd.DataFrame, type_column: str, method: str) -> pd.DataFrame:
    class_counts = df[type_column].value_counts()
    idx, idx_count, calc_samples, rebalance_method = configure_reablance_callables(method, class_counts)

    classes = df[type_column].unique()        
    critical_class = idx()
    critical_class_count = idx_count()
    
    non_critical_classes = [x for x in classes if x != critical_class]
    
    for non_critical_class in non_critical_classes:
        non_critical_class_count = class_counts[non_critical_class]
        n_samples = calc_samples(critical_class_count, non_critical_class_count)
        df = rebalance_method(df, type_column, non_critical_class, n_samples)
    return df
    
"""
Configures a SMOTE object based on a method.
Valid arguments for "method" parameter are: ["smote", "borderline-1", "borderline-2", "svm"]

Returns
-------
sm: SMOTE or BorderlineSMOTE1 or BorderlineSMOTE2 or SVMSMOTE
    The configured, specific SMOTE object
"""
def _configure_SMOTE(method: str):
    if (method == 'smote'):
        return SMOTE(sampling_strategy='not majority', random_state=42, n_jobs=_NUM_THREADS)
    elif (method in ['borderline-1', 'borderline-2']):
        return BorderlineSMOTE(sampling_strategy='not majority', random_state=42, n_jobs=_NUM_THREADS, kind=method)
    elif (method == 'svm'):
        raise NotImplementedError('svmSMOTE not implemented')
    else:
        raise ValueError('\"{}\" is invalid argument for method parameter\n\tValid arguments: [\"smote\", \"borderline-1\", \"borderline-2\", \"svm\"]'.format(method))

"""
Constructs a balanced DataFrame with correct labeling for synthetic data

Returns
-------
rebalanced_df: pandas.DataFrame
    The rebalanced DataFrame 
"""
def _construct_rebalanced_df(X_resampled: pd.DataFrame, Y_resampled: pd.DataFrame, type_column: str, original_df: pd.DataFrame) -> pd.DataFrame:
    datasets = original_df['datasetName']
    
    rebalanced_df = pd.DataFrame(data=X_resampled)
    rebalanced_df[type_column] = Y_resampled
    
    num_synthetic = len(rebalanced_df.index) - len(original_df.index)
    datasets = datasets.append(pd.Series(['SYNTHETIC'] * num_synthetic), ignore_index=True)
    
    rebalanced_df['datasetName'] = datasets
    
    return rebalanced_df

"""
Rebalances a DataFrame using a SMOTE method.
"df" argument should be unembedded data
Valid arguments for "method" parameter are: ["smote", "borderline-1", "borderline-2", "svm"]

Returns
-------
rebalanced_df: pandas.DataFrame
    The rebalanced DataFrame 
"""
def rebalance_SMOTE(df: pd.DataFrame, type_column: str, method: str, model_weights_path: str) -> pd.DataFrame:
    embedded_df = embed(df, type_column, model_weights_path)
    sm = _configure_SMOTE(method)
    
    X_resampled, Y_resampled = sm.fit_resample(embedded_df.drop(['datasetName',type_column], axis=1), embedded_df[type_column])
    
    return _construct_rebalanced_df(X_resampled, Y_resampled, type_column, df)
    