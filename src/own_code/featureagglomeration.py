"""
5. Use sklearn.cluster.FeatureAgglomeration to reduce the dimensionality of your data sets
and cluster it with your own k-Means, the one that you implemented in Work 1, and with the
AgglomerativeClustering from sklearn library. Compare your new results with the ones
obtained previously.
You will find useful information of how to deal with this algorithm at:
a. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import FeatureAgglomeration
from pathlib import Path
import matplotlib.pyplot as plt

# define path variables
# ROOT_PATH = Path(__file__).parent.parent.parent
# DATA_PATH = ROOT_PATH / 'data'
# RAW_DATA_PATH = DATA_PATH / 'raw'
# PROCESSED_DATA_PATH = DATA_PATH / 'processed'


class feature_agglomeration():

    def __init__(self,
               data: np.array):
        self.data = data
        self.labels = None
        self.transformed_data = None
        self.explained_variance_ratio = None

    def do_feature_agglomeration(self, num_clusters=None, affinity='euclidean', linkage='single',
                                 distance_threshold=None):
        x = self.data

        fa = FeatureAgglomeration(num_clusters, affinity=affinity, linkage=linkage,
                                  distance_threshold=distance_threshold)
        self.transformed_data = fa.fit_transform(x)

        self.labels = fa.labels_
        self.explained_variance_ratio = fa.explained_variance_ratio_

        return self.transformed_data

    def visualize(self, labels=None, axes=[0, 1, 2], figsize=(10, 10), original=True, save=False):
        if original:
            data = self.data
        else:
            data = self.transformed_data

        values = []
        for i in axes:
            values.append(data[:, i])
        values = np.array(values).T
        print(values.shape)
        dims = len(axes)

        if dims == 4:
            self.scatter_4D(values, labels, figsize)
        elif dims == 3:
            self.scatter_3D(values, labels, figsize)
        elif dims == 2:
            self.scatter_2D(values, labels, figsize)

        if save:
            plt.savefig(save)


    @staticmethod
    def scatter_4D(data, labels, figsize=(10, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=data[:, 3] * 10)
        plt.show()

    @staticmethod
    def scatter_3D(data, labels, figsize=(10, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=15)
        plt.show()

    @staticmethod
    def scatter_2D(data, labels, figsize=(10, 10)):
        fig = plt.figure(figsize=figsize)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='cool')
        plt.show()
