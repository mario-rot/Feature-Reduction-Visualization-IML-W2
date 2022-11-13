import sys
import os
sys.path.append('../../')
import argparse
from pathlib import Path
import numpy as np

from src.datapipeline import MLDataset
from src.algorithms.own_pca import OPCA
from src.algorithms.pca_sklearn import PCA_sklearn
from src.algorithms.featureagglomeration import feature_agglomeration
from src.algorithms.own_kmeans import OK_Means
from src.algorithms.agglomerative_sklearn import Agglomerative_sklearn
from src.algorithms.visualize_tsne import TSNE_sklearn

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-exp", "--Experiment", help = "['FRC':'Feature Reduction Comparison', 'CDR':'Clustering with Dimensionality Reduction']", default='FRC', type=str)
parser.add_argument("-ds", "--DataSet", help = "['iris', 'cmc', 'pen-based', 'vowel']", default='iris', type=str)
parser.add_argument("-dr", "--DimReduct", help = "['none','pca','feat_agg']", default='pca', type=str)
parser.add_argument("-cl", "--ClustAlg", help = "['kmeans','agg_cl']", default='kmeans', type=str)
parser.add_argument("-vt", "--VisTech", help = "['v_pca', 't-sne']", default='v_pca', type=str)
parser.add_argument("-save", "--Save", help = "Directory to save figs", default=False, type=str)
parser.add_argument("-nc", "--NClus", help = "Number of clusters (Exp 2)", default=3, type=int)
parser.add_argument("-nco", "--NComps", help = "Number of components", default=3, type=int)
parser.add_argument("-nv", "--NVis", help = "Number of axes to visualize", default=3, type=int)

args = parser.parse_args()

if args.Save:
    if not os.path.exists(args.Save):
        os.makedirs(args.Save)

data_path = Path('../../data/raw/' + args.DataSet + '.arff')
data =  MLDataset(data_path)
X = data.processed_data.values[:,:-1]
y = data.processed_data.values[:,-1]

params = {'DRT':args.DimReduct,
          'CA': args.ClustAlg,
          'VS': args.VisTech}
settings = {'pca': OPCA,
            'feat-agg': feature_agglomeration,
            'k-means': OK_Means,
            'agg-cl': Agglomerative_sklearn,
            'v-pca': OPCA,
            't-sne': TSNE_sklearn,
            2:[0,1],
            3:[0,1,2],
            4:[0,1,2,3],
            'vowel':'Vowel',
            'cmc':'CMC',
            'pen-based':'Pen-based'}

if args.Experiment == 'FRC':
########################### Experiment 1 ###################################
    args.Save = args.Save + settings[args.DataSet] + '/'

    # ----- Own PCA implementation
    opca = OPCA(X, args.DataSet)
    opca.fit(args.NComps)
    opca.visualize(y, settings[args.NVis], (7,7),original='Original', save=args.Save + '_opca_')
    opca.visualize(y, settings[args.NVis], (7,7),original='Transformed', save=args.Save + '_opca_')
    opca.visualize(y, settings[args.NVis], (7,7),original='Reconstructed',save=args.Save + '_opca_')
    opca.scree_plot(save=args.Save + '_opca_')

    # ----- Sklearn PCA implementation
    skpca = PCA_sklearn(X, args.DataSet)
    skpca.do_sklearn_PCA(args.NComps)
    skpca.visualize(y, settings[args.NVis], (7,7),original='Original', save=args.Save + '_skpca_')
    skpca.visualize(y, settings[args.NVis], (7,7),original='Transformed', save=args.Save + '_skpca_')
    skpca.visualize(y, settings[args.NVis], (7,7),original='Reconstructed',save=args.Save + '_skpca_')
    skpca.scree_plot(save=args.Save + '_skpca_')

    # ----- Sklearn Incremental PCA implementation
    skipca = PCA_sklearn(X, args.DataSet)
    skipca.do_sklearn_incrementalPCA(args.NComps)
    skipca.visualize(y, settings[args.NVis], (7,7),original='Original', save=args.Save + '_skipca_')
    skipca.visualize(y, settings[args.NVis], (7,7),original='Transformed', save=args.Save + '_skipca_')
    skipca.visualize(y, settings[args.NVis], (7,7),original='Reconstructed',save=args.Save + '_skipca_')
    skipca.scree_plot(save=args.Save + '_skipca_')

    # ----- Sklearn Feature Agglomeration implementation
    fa = feature_agglomeration(X, args.DataSet)
    fa.fit(args.NComps)
    fa.visualize(y, settings[args.NVis], (7,7),original='Original', save=args.Save + '_fa_')
    fa.visualize(y, settings[args.NVis], (7,7),original='Transformed', save=args.Save + '_fa_')
    fa.visualize(y, settings[args.NVis], (7,7),original='Reconstructed',save=args.Save + '_fa_')

else: 
########################### Experiment 2 ###################################

    # ----- Dimensionality Reduction
    dimen_red = settings[params['DRT']](X,  args.DataSet)
    transformed_data = dimen_red.fit(args.NComps)
    print('# ----- Dimensionality Reduction - Done!')

    # ----- Clustering
    clusters_dr = settings[params['CA']](transformed_data)
    labels_dr = clusters_dr.fit(args.NClus)
    print('# ----- Transformed Data Clustering - Done!')

    clusters_wdr = settings[params['CA']](X)
    labels_wdr = clusters_wdr.fit(args.NClus)
    print('# ----- Original Data Clustering - Done!')

    #----- Visualization
    plt_name = '../../figures/Test2_2d/'+ settings[args.DataSet] +'/' + args.DataSet + '_' + args.DimReduct + '_' + args.ClustAlg + '_' + args.VisTech 

    visualize_dr = settings[params['VS']](transformed_data,  args.DataSet)
    visualize_dr.fit(args.NVis)
    visualize_dr.visualize(labels_dr, settings[args.NVis], (7,7),original='Transformed', save=plt_name+'_dr_')
    print('# ----- Transformed Data Visualization - Done!')

    visualize_wdr = settings[params['VS']](X,  args.DataSet)
    visualize_wdr.fit(args.NVis)
    visualize_wdr.visualize(labels_wdr, settings[args.NVis], (7,7),original='Transformed', save=plt_name+'_wdr_')
    print('# ----- Original Data Visualization - Done!')