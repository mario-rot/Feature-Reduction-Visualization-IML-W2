"""
6. Visualize in low-dimensional space. You need to visualize your original data sets, the result of
the k-Means and Agglomerative clustering algorithms without the dimensionality reduction,
and the result of the k-Means and Agglomerative clustering algorithms with the dimensionality
reduction. To visualize in a low-dimensional space (2D or 3D) you will use: PCA and t-SNE.
You will find useful information of how to deal with this algorithm at:
a. https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
b. https://www.datacamp.com/community/tutorials/introduction-t-sne
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




class TSNE_sklearn():

    def __init__(self,
               data: np.array):
        self.data = data
        self.labels = None
        self.transformed_data = None
        self.explained_variance_ratio = None

    def do_tsne(self, num_components=None, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        tsne = TSNE(num_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, init='random')
        self.transformed_data = tsne.fit_transform(self.data)
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
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='cool')
        plt.show()
