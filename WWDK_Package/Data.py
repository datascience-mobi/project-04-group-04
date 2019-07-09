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

    
     def process(self, method="tsne"): # Data preprocessing
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

        data_nonzero_variance = ar_data[columns[ar_data.var() != 0]]  
        normalized_data = preprocessing.normalize(data_nonzero_variance)
        scaled_data = preprocessing.scale(normalized_data)
        pca = PCA()
        pca_data = pca.fit_transform(scaled_data)

        if method == "tsne":
            tsned = TSNE()
            processed_data = tsned.fit_transform(pca_data)
        elif method == "umap":
            umaped = umap.UMAP()
            processed_data = umaped.fit_transform(pca_data)
        else:
            print("No valid method!")

        return processed_data, ar_data, columns, rows
   