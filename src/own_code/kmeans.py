from typing import Union, List
import numpy as np
import pandas as pd
from collections import namedtuple


class OK_Means():
    def __init__(self, 
                 data: np.array):
        self.data = data
        self.n_instances = data.shape[0] 
        self.n_features = data.shape[1]

    def kmeans(self,
               n_clusters: int,
               init_centers: np.ndarray,
               max_iteration: int,
               verbose: bool = False):
        # Data
        df = pd.DataFrame(self.data)

        # Init. parameters
        centers = None
        labels = None
        inertia = None
        new_centroids = np.zeros((n_clusters, self.n_features))
        ii = 0


        while ii < max_iteration:
            if ii == 0:
                centers = init_centers
            else:
                centers = new_centroids

            # Distances
            distances = self.distance_matrix(self,self.data, centers)

            # Obtain Labels
            labels = np.array([np.argmin(sample) for sample in distances])

            df['labels'] = labels
            new_centroids = df.groupby('labels').mean().values

            # Get inertia
            inertia = self.get_inertia(distances, labels, self.n_instances)

            if verbose:
                print(f'Iteration {ii}, Inertia: {inertia}')

            # Convergence
            if np.all(new_centroids == centers):
                centers = new_centroids
                if verbose:
                    print(f'Converged iteration {ii + 1}, Inertia: {inertia}')
                break
            ii += 1

        return centers, labels, inertia


    def kmeans_exp(self,
                   n_clusters: int,
                   max_iteration=300,
                   n_attempt=10,
                   verbose=False):

        # Result class
        Results = namedtuple('Results', 'labels centers inertia')
        # Data
        n_samples, n_features = self.n_instances, self.n_features

        # Init parameters
        best_inertia = None
        results = None
        best_attempt = None

        for i in range(n_attempt):
            if verbose:
                print(f'-----Attempt {i + 1}-----')

            # Initial random centroids from samples
            ii = np.random.choice(n_samples, size=n_clusters)
            random_centroids = self.data[ii]

            # Run kmeans
            centers, labels, inertia = self.kmeans(n_clusters,
                                                  random_centroids,
                                                  max_iteration,
                                                  verbose)

            # Select best inertia
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_attempt = i
                results = Results(labels=labels,
                                centers=centers,
                                inertia=inertia)
        if verbose:
            print(f'Best attempt {best_attempt + 1}')

        return results
   

    @staticmethod
    def squared_dist(a: np.ndarray, b: np.ndarray) -> float:
        return np.square(a - b).sum()
    @staticmethod
    def distance_matrix(self,XX: np.ndarray, YY: np.ndarray):
        distances = []
        for x in XX:
            for y in YY:
                distances.append(self.squared_dist(x, y))
        return np.array(distances).reshape(XX.shape[0], -1)
    @staticmethod
    def get_inertia(distances: np.ndarray,
                    labels: Union[List, np.ndarray],
                    n_samples: int):
        ll = np.array(
            [distances[i][labels[i]] for i in range(n_samples)])
        return ll.sum()
