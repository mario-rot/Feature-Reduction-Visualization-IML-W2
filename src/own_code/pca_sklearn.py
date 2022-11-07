"""
3. Analysis three data sets using PCA and IncrementalPCA from sklearn
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from pathlib import Path
import matplotlib.pyplot as plt

colors = plt.cm.Set2.colors
# define path variables
# ROOT_PATH = Path(__file__).parent.parent.parent
# DATA_PATH = ROOT_PATH / 'data'
# RAW_DATA_PATH = DATA_PATH / 'raw'
# PROCESSED_DATA_PATH = DATA_PATH / 'processed'


class PCA_sklearn():

    def __init__(self,
               data: np.array):
        self.data = data
        self.n_features = data.shape[1]
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None
        self.explained_variance_ratio = None

    def do_sklearn_PCA(self, num_components=None):
        x = self.data

        pca = PCA(num_components)
        self.transformed_data = pca.fit_transform(x)

        # Explained Variance = eigenvalues
        self.eigenvalues = pca.explained_variance_

        # Eigenvectors
        self.eigenvectors = pca.components_

        self.explained_variance_ratio = pca.explained_variance_ratio_

        return self.transformed_data

    def do_sklearn_incrementalPCA(self, num_components):
        x = self.data

        ipca = IncrementalPCA(num_components)
        self.transformed_data = ipca.fit_transform(x)

        # Explained Variance = eigenvalues
        self.eigenvalues = ipca.explained_variance_

        # Eigenvectors
        self.eigenvectors = ipca.components_

        self.explained_variance_ratio = ipca.explained_variance_ratio_

        return self.transformed_data

    def visualize(self, labels, axes=[0, 1, 2], figsize=(10, 10), original=True, save=False):
        if original:
            data = self.data
        else:
            data = self.transformed_data
            print(data.shape)

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

    def scree_plot(self, save=False):
        plt.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
        plt.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')

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

