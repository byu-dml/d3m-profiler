from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

_NUM_THREADS = cpu_count() - 1

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
def _configure_SMOTE(method: str, **opts):
    k_neighbors = opts.get('k_neighbors', 5)
    if (method == 'smote'):
        return SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=k_neighbors, n_jobs=_NUM_THREADS)
    elif (method in ['borderline-1', 'borderline-2']):
        return BorderlineSMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=k_neighbors, n_jobs=_NUM_THREADS, kind=method)
    elif (method == 'svm'):
        raise NotImplementedError('svmSMOTE not implemented')
    else:
        raise ValueError('\"{}\" is invalid argument for method parameter\n\tValid arguments: [\"smote\", \"borderline-1\", \"borderline-2\", \"svm\"]'.format(method))


"""
Rebalances a DataFrame using a SMOTE method.
"df" argument should be unembedded data
Valid arguments for "method" parameter are: ["smote", "borderline-1", "borderline-2", "svm"]

Returns
-------
rebalanced_df: pandas.DataFrame
    The rebalanced DataFrame 
"""
def rebalance_SMOTE(X_embedded, y, method: str):
    k_neighbors = min(np.unique(y, return_counts=True)[1]) - 1  # number of neighbors not including self
    if k_neighbors < 1:
        raise ValueError(f'Not enough data to rebalance. K-Neighbors must be 1 or more (is {k_neighbors}).')
    sm = _configure_SMOTE(method, k_neighbors=k_neighbors)
    return sm.fit_resample(X_embedded, y)
