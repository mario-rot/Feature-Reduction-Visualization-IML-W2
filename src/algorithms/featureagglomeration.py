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
               data: np.array, 
               data_name: str):
        self.data = data
        self.data_name = data_name
        self.labels = None
        self.transformed_data = None
        self.explained_variance_ratio = None

    def fit(self, num_clusters=None, affinity='euclidean', linkage='single',
                                 distance_threshold=None):
        x = self.data

        fa = FeatureAgglomeration(num_clusters, affinity=affinity, linkage=linkage,
                                  distance_threshold=distance_threshold)
        self.transformed_data = fa.fit_transform(x)

        self.reconstructed_data = fa.inverse_transform(self.transformed_data)

        self.labels = fa.labels_

        return self.transformed_data

    def visualize(self, labels, axes=[0, 1, 2], figsize=(10, 10), original=True, save=None):
        if original == 'Original':
            data = self.data
            title = 'Original Data'
        elif original == 'Reconstructed':
            data = self.reconstructed_data
            title ='Reconstructed Data'
        else:
            data = self.transformed_data
            title = 'Transformed Data'

        values = []
        for i in axes:
            values.append(data[:, i])
        values = np.array(values).T
        dims = len(axes)

        if dims == 4:
            self.scatter_4D(values, labels, axes, title, self.data_name, figsize, save)
        elif dims == 3:
            self.scatter_3D(values, labels, axes, title, self.data_name, figsize, save)
        elif dims == 2:
            self.scatter_2D(values, labels, axes, title, self.data_name, figsize, save)

        if save:
            plt.savefig(save)

    @staticmethod
    def scatter_4D(data, labels, axes, title, data_name, figsize=(10, 10), save=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=data[:, 3] * 10)
        ax.set_title(title)
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_zlabel(axes[2])

        if save:
            ax.savefig(save + 'scatter_plot_4D_{}_{}.pdf'.format(data_name, title))
        
        plt.show()


    @staticmethod
    def scatter_3D(data, labels, axes, title, data_name, figsize=(10, 10), save=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=15)
        ax.set_title(title)
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.set_zlabel(axes[2])

        if save: 
            fig.savefig(save + 'scatter_plot_3D_{}_{}.pdf'.format(data_name, title))
        
        plt.show()


    @staticmethod
    def scatter_2D(data, labels, axes, title, data_name, figsize=(10, 10), save=None):
        fig = plt.figure(figsize=figsize)
        plt.scatter(data[:, 0], data[:, 1], c=labels) #, cmap='cool' )
        plt.title(title)
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])

        if save:
            plt.savefig(save + 'scatter_plot_2D_{}_{}.pdf'.format(data_name, title))

        plt.show()
