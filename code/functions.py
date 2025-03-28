import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def clean_data(df, standardize=False, fill_type=None, one_hot=False):
    """ Fill missing values, standardize, one-hot encode """

    assert fill_type in [None, 'simple','knn']
    
    # Drop columns
    x1 = df.drop(columns=['C_02']) # remove C_02, since it is a constant

    # Fill missing values
    if fill_type == 'simple': # fillna: median, mode, mean
        x1['C_01'] = x1['C_01'].fillna(x1['C_01'].median())  # median
        x1['C_03'] = x1['C_03'].fillna(x1['C_03'].median())  # median
        x1['C_05'] = x1['C_05'].fillna(x1['C_05'].median())  # median
        x1['C_04'] = x1['C_04'].fillna(x1['C_04'].mode()[0]) # mode
        x1 = x1.fillna(x1.mean()) # mean

    elif fill_type == 'knn': # fillna: KNN
        col_names = x1.columns
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        x1 = imputer.fit_transform(x1)
        x1[:, -5:] = np.round(x1[:, -5:]) # round categorical features
        x1 = pd.DataFrame(x1, columns=col_names)
    elif fill_type == None:
        pass
    if standardize:
        x1 = (x1 - x1.mean()) / x1.std()
    
    if one_hot:
        x1 = pd.get_dummies(x1, columns=[f'C_0{n}' for n in range(1, 6) if n!=2])
    
    return x1


def bootstrap(X, N=None):
    """ Bootstrap resampling """
    if N is None:
        N = X.shape[0]
    idx = np.random.randint(0, X.shape[0], N)
    X_boot = X.iloc[idx, :]
    return X_boot