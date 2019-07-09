import random
import math
from statistics import mean
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import urllib.request as url
import numpy as np
import scanpy as sc
import pandas as pd
import tarfile
import csv
from numba import njit, jit
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class Loader(): # Input:dData, Output: processed dataset ready for kmeans
    
    """Encapsulates data for clustering."""
    def __init__(self, page):
        self.page = tarfile.open(url.urlretrieve(page, filename=None)[0]).extractall()
        self.source = './filtered_gene_bc_matrices/hg19/'
        self.matrix = sc.read_10x_mtx(self.source, var_names='gene_symbols', cache=True)
        self.data = self.page

    
    def process(self, method="tsne", pca = True): # Data preprocessing
        self.matrix.var_names_make_unique()
        ar_data = self.matrix.to_df()
        columns = []
        with open(self.source + "genes.tsv") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                columns.append(row[1])
        rows = []
        with open(self.source + "barcodes.tsv") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                rows.append(row[0]) 

        return ar_data, columns, rows

    def _pca(self): # Principal component analysis
        #scaled = preprocessing.scale(self._data)
        pca = PCA()
        return pca.fit_transform(self.data)
        #per_var = np.round(pca.explained_variance_ratio_*100, decimals= 1)
        #pca_data = pca.transform(self-_data)
        #labels = ["PC" + str(x) for x in range(1,len(per_var)+1)]
     #t-distributed stochastic neighbor embedding
    
    def to_matrix(self):                                # converts to scanpy matrix 
        return self.matrix

    def _to_df(self):                                 # converts data to pandas dataframe
        self.data = self.matrix.to_df()

        return self.data
        
   