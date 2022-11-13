import sys
from time import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import src.evaluation_metrics as em

class Agglomerative_sklearn():
    def __init__(self, 
                 data: np.array):
        self.data = data
        self.n_instances = data.shape[0] 
        self.n_features = data.shape[1]

    def fit(self,
                      n_clusters: int,
                      affinity:str = 'euclidean',
                      linkage:str = 'average'):
        clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                             affinity=affinity,
                                             linkage=linkage).fit(self.data)   
        return clustering.labels_
    
    def agglo_exp(self,
                  y_true: np.ndarray,
                  parameters: Dict,
                  internal_metrics: em.INTERNAL_METRICS.values(),
                  external_metrics = em.EXTERNAL_METRICS.values()):
        X = self.data
        t0 = time()
        agglo_clustering = []
        da = []
        for affinity in parameters['affinity']:
            for linkage in parameters['linkage']:
                for k in parameters['n_clusters']:
                    t0 = time()
                    # Perform clustering
                    clustering = AgglomerativeClustering(n_clusters=k,
                                                        affinity=affinity,
                                                        linkage=linkage).fit(X)
                    tf = time() - t0
                    agglo_clustering.append(clustering)
                    # Save in a list
                    result = [tf, affinity, linkage, k]
                    # Internal index metrics
                    result += [m(X, clustering.labels_)
                            for m in internal_metrics]
                    # External index metrics
                    result += [m(y_true, clustering.labels_)
                            for m in external_metrics]
                    da.append(result)

        func_time = time() - t0
        return da, agglo_clustering, func_time


    def agglo_evaluation(self,
                         data_name: str,
                         n_clusters: int,
                         y_true: np.array,
                         save: bool = False):
        # Data
        X = self.data

        # Parameters for cluster
        params = {'affinity': ['euclidean', 'cosine'],
                'linkage': ['single', 'complete', 'average'],
                'n_clusters': n_clusters}

        # Perform sensitive analysis
        da = self.agglo_exp(X, y_true, params)
        metric_data, clus, global_time = da
        columns = ['time', 'affinity', 'linkage', 'n_clusters']
        columns = columns + list(definitions.INTERNAL_METRICS.keys()) + list(
            definitions.EXTERNAL_METRICS.keys())

        # Metric dataset
        metric_df = pd.DataFrame(metric_data, columns=columns)

        if verbose:
            print(metric_df)

        if save:
            metric_df.to_csv(PROCESSED_DATA_PATH / f'agglo_results_{data_name}')

        return metric_df, global_time


