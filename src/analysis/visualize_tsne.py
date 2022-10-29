"""
6. Visualize in low-dimensional space. You need to visualize your original data sets, the result of
the k-Means and Agglomerative clustering algorithms without the dimensionality reduction,
and the result of the k-Means and Agglomerative clustering algorithms with the dimensionality
reduction. To visualize in a low-dimensional space (2D or 3D) you will use: PCA and t-SNE.
You will find useful information of how to deal with this algorithm at:
a. https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
b. https://www.datacamp.com/community/tutorials/introduction-t-sne
"""

from sklearn.manifold import TSNE
