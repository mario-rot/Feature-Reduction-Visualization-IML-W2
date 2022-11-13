import sys
import os
sys.path.append('../../')
import argparse
from pathlib import Path
import numpy as np
import timeit

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

print('############################################# Dataset Statistis #######################################################')
print('\t\t#-------------- Dataset Statistis - RAW DATA -')
print(data.statistics('raw'))
print('\t\t#-------------- Dataset Statistis - PROCESSED DATA -')
print(data.statistics('processed'))

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
    print('\t\t#-------------- DOING EXPERIMENT 1 ({}) DATASET-'.format(settings[args.DataSet]))
    # args.Save = args.Save + settings[args.DataSet] + '/' # Uncomment to save properly plots

    # ----- Own PCA implementation
    startopca = timeit.default_timer()
    opca = OPCA(X, args.DataSet)
    opca.fit(args.NComps)
    stopopca = timeit.default_timer()
    print('\n\nTime OPCA: {}\n\n'.format(stopopca - startopca))  
    print('\t# ----- Own PCA implementation')
    print('\t# - Covariance Matrix -')
    print(opca.cov_mat)
    print('\t# - Eigenvalues Unordered -')
    print(opca.U_eigenvalues)
    print('\t# - Eigenvectors Unordered -')
    print(opca.U_eigenvectors)
    print('\t# - Eigenvalues Ordered -')
    print(opca.eigenvalues)
    print('\t# - Eigenvectors Ordered -')
    print(opca.eigenvectors)
    opca.visualize(y, settings[args.NVis], (7,7),original='Original') #, save=args.Save + '_opca_') # Uncomment last part to save properly plots
    opca.visualize(y, settings[args.NVis], (7,7),original='Transformed') #, save=args.Save + '_opca_') # Uncomment last part to save properly plots
    opca.visualize(y, settings[args.NVis], (7,7),original='Reconstructed') #,save=args.Save + '_opca_') # Uncomment last part to save properly plots
    opca.scree_plot() #save=args.Save + '_opca_') # Uncomment last part to save properly plots

    # ----- Sklearn PCA implementation
    startskpca = timeit.default_timer()
    skpca = PCA_sklearn(X, args.DataSet)
    stopskpca = timeit.default_timer()
    print('\n\n Time SKPCA: ', stopskpca - startskpca) 
    skpca.do_sklearn_PCA(args.NComps)
    skpca.visualize(y, settings[args.NVis], (7,7),original='Original') #, save=args.Save + '_skpca_') # Uncomment last part to save properly plots
    skpca.visualize(y, settings[args.NVis], (7,7),original='Transformed') #, save=args.Save + '_skpca_') # Uncomment last part to save properly plots
    skpca.visualize(y, settings[args.NVis], (7,7),original='Reconstructed') #,save=args.Save + '_skpca_') # Uncomment last part to save properly plots
    skpca.scree_plot() # save=args.Save + '_skpca_') # Uncomment last part to save properly plots

    # ----- Sklearn Incremental PCA implementation
    skipca = PCA_sklearn(X, args.DataSet)
    skipca.do_sklearn_incrementalPCA(args.NComps)
    skipca.visualize(y, settings[args.NVis], (7,7),original='Original') #, save=args.Save + '_skipca_') # Uncomment last part to save properly plots
    skipca.visualize(y, settings[args.NVis], (7,7),original='Transformed') #, save=args.Save + '_skipca_') # Uncomment last part to save properly plots
    skipca.visualize(y, settings[args.NVis], (7,7),original='Reconstructed') #,save=args.Save + '_skipca_') # Uncomment last part to save properly plots
    skipca.scree_plot() #save=args.Save + '_skipca_') # Uncomment last part to save properly plots

    # ----- Sklearn Feature Agglomeration implementation
    fa = feature_agglomeration(X, args.DataSet)
    fa.fit(args.NComps)
    fa.visualize(y, settings[args.NVis], (7,7),original='Original') #, save=args.Save + '_fa_') # Uncomment last part to save properly plots
    fa.visualize(y, settings[args.NVis], (7,7),original='Transformed') #, save=args.Save + '_fa_') # Uncomment last part to save properly plots
    fa.visualize(y, settings[args.NVis], (7,7),original='Reconstructed') #,save=args.Save + '_fa_') # Uncomment last part to save properly plots

else: 
########################### Experiment 2 ###################################
    print('\t\t#-------------- DOING EXPERIMENT 2 ({}) DATASET-'.format(settings[args.DataSet]))
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
    # plt_name = '../../figures/Test2_2d/'+ settings[args.DataSet] +'/' + args.DataSet + '_' + args.DimReduct + '_' + args.ClustAlg + '_' + args.VisTech # Uncomment to save properly plots

    visualize_dr = settings[params['VS']](transformed_data,  args.DataSet)
    visualize_dr.fit(args.NVis)
    visualize_dr.visualize(labels_dr, settings[args.NVis], (7,7),original='Transformed') #, save=plt_name+'_dr_') # Uncomment last part to save properly plots
    print('# ----- Transformed Data Visualization - Done!')

    visualize_wdr = settings[params['VS']](X,  args.DataSet)
    visualize_wdr.fit(args.NVis)
    visualize_wdr.visualize(labels_wdr, settings[args.NVis], (7,7),original='Transformed') #, save=plt_name+'_wdr_') # Uncomment last part to save properly plots
    print('# ----- Original Data Visualization - Done!')