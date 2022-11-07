"""
3. Analysis three data sets using PCA and IncrementalPCA from sklearn
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
# import matplotlib.pyplot as plt

# define path variables
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / 'data'
RAW_DATA_PATH = DATA_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'


def do_sklearn_PCA(data_name, num_components=None):
    data_name = 'iris.csv'
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1].values
    # X = df.iloc[:, :-1]
    # X = df.loc[:, df.keys()]
    Y = df['y_true']

    pca = PCA(num_components)
    pca.fit_transform(X.T)
    #components = pca.fit_transform(X)

    print(X.shape)
    print((pca.components_).shape)
    return pca.components_, Y


def do_sklearn_incrementalPCA():
    return


components, Y = do_sklearn_PCA('iris.csv', 2)
