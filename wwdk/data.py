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
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
from sklearn.datasets import load_digits

class Loader(): # Input:dData, Output: processed dataset ready for kmeans
    
    """Encapsulates data for clustering."""
    def __init__(self, page):
        self.page = tarfile.open(url.urlretrieve(page, filename=None)[0]).extractall()
        self.source = './filtered_gene_bc_matrices/hg19/'
        self.matrix = sc.read_10x_mtx(self.source, var_names='gene_symbols', cache=True)
        self.data = self.page

    
    def process(self, method="umap"): #Data preprocessing
        self.matrix.var_names_make_unique()
        ar_data = self.matrix.to_df()
        columns = []
        with open(self.source + "genes.tsv") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                columns.append(row[1])
        columns = np.array(columns)
        rows = []
        with open(self.source + "barcodes.tsv") as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                rows.append(row[0]) 
        rows = np.array(rows)   
        
        ar_data_without_mt = ar_data[np.sum(ar_data[ar_data.columns[ar_data.columns.str.startswith("MT-") == True]],axis=1)/np.sum(ar_data,axis=1) <= 0.05]
        # Those cells are deleted, which have a fraction of counts mito genes vs. all genes higher 5%. 
        ar_data_without_mt_bigger_three = ar_data_without_mt.T[(ar_data_without_mt.astype(bool).sum(axis=0) > 3)].T
        # Those genes are deleted, which are expressed in less than three cells.

        normalized_data = preprocessing.normalize(ar_data_without_mt_bigger_three)
        
        pca = PCA(n_components = 40)
        pca_data = pca.fit_transform(normalized_data)
        per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
        labels = ["PC" + str(x) for x in range(1,len(per_var)+1)]
        pca_df = pd.DataFrame(pca_data, columns= labels)

        if method == "tsne":
            processed_data = TSNE().fit_transform(pca_df)
        elif method == "umap":
            processed_data = umap.UMAP(n_neighbors=10).fit_transform(pca_df)
        else:
            print("No valid method!")

        return processed_data, ar_data_without_mt_bigger_three, ar_data, columns, rows
   