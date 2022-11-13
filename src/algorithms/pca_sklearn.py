"""
3. Analysis three data sets using PCA and IncrementalPCA from sklearn
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "tab20"
# Para cambiar el ciclo de color por defecto en Matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
#Set_ColorsIn(plt.cm.Set2.colors)
colors = plt.cm.tab20.colors


class PCA_sklearn():

    def __init__(self,
               data: np.array,
               data_name: str):
        self.data = data
        self.data_name = data_name
        self.n_features = data.shape[1]
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None
        self.explained_variance_ratio = None
        self.reconstructed_data = None
        self.Version = None

    def do_sklearn_PCA(self, num_components=None):
        self.Version='sklearn PCA'
        x = self.data

        pca = PCA(num_components)
        self.transformed_data = pca.fit_transform(x)

        # Explained Variance = eigenvalues
        self.eigenvalues = pca.explained_variance_

        # Eigenvectors
        self.eigenvectors = pca.components_

        self.explained_variance_ratio = pca.explained_variance_ratio_

        # Reconstructed data
        self.reconstructed_data = pca.inverse_transform(self.transformed_data)

        return self.transformed_data

    def do_sklearn_incrementalPCA(self, num_components):
        self.Version='Incremental PCA'
        x = self.data

        ipca = IncrementalPCA(num_components)
        self.transformed_data = ipca.fit_transform(x)

        # Explained Variance = eigenvalues
        self.eigenvalues = ipca.explained_variance_

        # Eigenvectors
        self.eigenvectors = ipca.components_

        self.explained_variance_ratio = ipca.explained_variance_ratio_

        # Reconstructed Data
        self.reconstructed_data = ipca.inverse_transform(self.transformed_data)


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
            self.scatter_4D(values, labels, axes, title+ ' ' + self.Version, self.data_name,  figsize, save)
        elif dims == 3:
            self.scatter_3D(values, labels, axes, title+ ' ' + self.Version, self.data_name, figsize, save)
        elif dims == 2:
            self.scatter_2D(values, labels, axes, title + ' ' + self.Version, self.data_name, figsize, save)

        if save:
            plt.savefig(save)
        
    def scree_plot(self, save=False):
        plt.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
        plt.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
        plt.title('Explained Variance {}'.format(self.Version))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')

        if save:
            plt.savefig(save + 'scree_plot_{}.pdf'.format(self.data_name))

        plt.show()


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

