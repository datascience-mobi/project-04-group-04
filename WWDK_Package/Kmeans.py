

import random
import math
from statistics import mean
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import urllib.request as url
import numpy as np
import scanpy as sc
import pandas as pd
import tarfile
import csv
from numba import njit, jit
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
class Kmeans(BaseEstimator, ClusterMixin, TransformerMixin):               # Input: processed dataset, Output: clustered data (kmeans, kmeans++)
    def __init__(self, inits=10, k=8, maxit=300):
        
        self.labels_ = None
        self.cluster_centers_ = None
        self._inits = inits
        self._k = k
        self._maxit = maxit
       # dot = np.random.choice(range(len(self._data)), self._k, replace=False)
        #self._clusters = self._data[dot]
   

    def fit(self,data):
        self._data = data
        best_clust = float('inf')
        
        for i in (range(self._inits)):
            dot = np.random.choice(range(len(self._data)), self._k, replace=False)
            self.cluster_centers_ = self._data[dot]
            for i in range(self._maxit):
                clusters = np.expand_dims(self.cluster_centers_, axis=1)
                data = np.expand_dims(self._data, axis=0)
                eucl = np.linalg.norm(clusters-data, axis=2) # euclidean dist by using integrated numpy function
                self.labels_ = np.argmin(eucl, axis = 0)
                for i in range(self._k): # range of clusters
                    position = np.where(self.labels_ == i) # position im array bestimmen und dann die entspechenden punkte aus data auslesen
                    self.cluster_centers_[i] = self._data[position].mean(axis = 0)
                    #out = pd.DataFrame(data[np.argwhere(dist == i)].squeeze())
                overall_quality = np.sum(np.min(eucl.T, axis=1))
                if overall_quality < best_clust:
                    best_clust = overall_quality
                    best_dist = self.labels_
                    best_centers = self.cluster_centers_
            self.cluster_centers_ = best_centers
            self.labels_ = best_dist
                
        return self
   
    
    def predict(self, X):
        clusters = np.expand_dims(self.cluster_centers_, axis=1)
        data = np.expand_dims(X, axis=0)
        eucl = np.linalg.norm(clusters-data, axis=2) # euclidean dist by using integrated numpy function
        self.labels_ = np.argmin(eucl, axis = 0)
        return self.labels_ #returns the cluster with minimum distance
    
    def transform(self, X):
        clusters = np.expand_dims(self.cluster_centers_, axis=1)
        data = np.expand_dims(X, axis=0)
        eucl = np.linalg.norm(clusters-data, axis=2)
        return eucl.T
        
    '''
        #@jit(nopython=True)
    def fit(self):
        for i in range(k): # range of clusters
            position = np.where(self._dist == i) # position im array bestimmen und dann die entspechenden punkte aus data auslesen
            self._clusters[i] = self._data[position].mean(axis = 0)
                #out = pd.DataFrame(data[np.argwhere(dist == i)].squeeze())
            return self._clusters # return new clusters  
    def kmeans(self, inits=None, k=None, maxit=None):   # the original
        if inits is None:
            inits = self._inits
        if k is None:
            k = self._k
        if maxit is None:
            maxit = self._maxit
    
        clusters = create_clusters(self._data, k)
        for i in range(maxit):
            dist = distances(clusters,self._data)
            clusters = fit(self._data,dist,k)
        
        self._clusters = clusters
        self._dist = dist
        '''