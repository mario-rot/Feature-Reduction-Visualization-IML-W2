import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as SLA
import itertools

data_path = '../data/processed/'

class OPCA():
    def __init__(self, 
                 data: np.array):
        self.data = data
        self.n_instances = data.shape[0] 
        self.n_features = data.shape[1]
        self.means = None
        self.cov_mat = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None
        self.reconstructed_data = None

    def fit(self, axes = 2):
        # Computing mean of each axis(feature) in the data
        self.means = self.data.mean(0)
        # Substracting each data point from the means
        sub_means = np.subtract(self.data,self.means)
        # Computing covariance matrix 
        self.cov_mat = self.covariance_matrix(sub_means)
        # Getting eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = SLA.eig(self.cov_mat, left = True, right = False)
        # values = values.astype(float)
        # Descending ordering to eigenvalues
        order = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[order]
        # Ordering eigenvectors according to eigenvalues
        self.eigenvectors = self.eigenvectors.T[order]
        self.eigenvectors = np.array([-self.eigenvectors[i] if i%2 != 0 else self.eigenvectors[i] for i in range(len(self.eigenvectors))])
        # Choosing number of components of the feature vector
        feat_vec = self.eigenvectors[:axes]
        # Transforming the data
        self.transformed_data = (feat_vec.dot(sub_means.T))
        # Reconstructing to the original data
        self.reconstructed_data = self.transformed_data.T.dot(feat_vec)+self.means
        return -self.transformed_data.T

    @staticmethod
    def covariance_matrix(sub_means):
        # Getting the shape of the matrix
        cov = np.zeros((sub_means.shape[1], sub_means.shape[1]))
        # Computing all possible features combinations
        c = list(itertools.product(sub_means.T,repeat = 2))
        s = int(sub_means.shape[1])
        # Computing the matrix
        for k,i in enumerate(c):
            cov[(-(k%s-(k))//s,k%s)] = (i[0]*i[1]).sum()/(sub_means.shape[0]-1)
        return cov
