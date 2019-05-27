
import matplotlib.pyplot as plt
import urllib.request as url
import numpy as np
import scanpy as sc
import pandas as pd
import tarfile
import csv
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random
import math
from statistics import mean
from sklearn.datasets.samples_generator import make_blobs
class Kmeans():               # Input: processed dataset, Output: clustered data (kmeans, minibatch, kmeans++)
    def __init__(self, data, inits=10, k=8, maxit=300):
        self._data = data
        self._clusters = 0
        self._dist = 0
        self._inits = inits
        self._k = k
        self._maxit = maxit
    def kmeans(self, inits=None, k=None, maxit=None):   # the original
        if inits is None:
            inits = self._inits
        if k is None:
            k = self._k
        if maxit is None:
            maxit = self._maxit
    
        def create_clusters(data,k):
            dot = np.random.choice(range(len(data)), k, replace=False)
            return data[dot]
        
        def distances(clusters,data):
            clusters = np.expand_dims(clusters, axis=1)
            data = np.expand_dims(data, axis=0)
            eucl = np.linalg.norm(clusters-data, axis=2) # euclidean dist by using integrated numpy function
    
            return np.argmin(eucl, axis = 0) #returns the cluster with minimum distance
    
        def fit(data,dist,k):
            for i in range(k): # range of clusters
                position = np.argwhere(dist == i) # position im array bestimmen und dann die entspechenden punkte aus data auslesen
                clusters[i] = data[position].mean(axis = 0)
                #out = pd.DataFrame(data[np.argwhere(dist == i)].squeeze())
            return clusters # return new clusters
    
        clusters = create_clusters(self._data, k)
        for i in range(maxit):
            dist = distances(clusters,self._data)
            clusters = fit(self._data,dist,k)
        
        self._clusters = clusters
        self._dist = dist
        return self._clusters
        
        
    def kmeans2plus(self, inits, cluster, maxit):  # kmeans++
        pass
    def mbkmeans(self, inits, cluster, maxit): # mini batch kmeans faster but less precicse(?)
        pass
    def multikmeans(self, method, inits, cluster, maxit): # makes multiple kmeas of preferred type to show difference
        pass
    
    def plot(self): 
        try:
            for i in range(self._k):
                graph = pd.DataFrame(self._data[np.argwhere(self._dist == i)].squeeze())
                center = pd.DataFrame(self._clusters[i]).T
                #print("Cluster"+ str(i) +  " -- Assigned Points \n" + str(graph))
                plt.plot(graph[0], graph[1], "o")
                plt.plot(center[0],center[1], "kx")
            plt.show
        except TypeError:
            print("Run a clustering method first!")
class DataLoader(): # Input:dData, Output: processed dataset ready for kmeans
    
    def __init__(self, data):
        self._data = data
    def url_extract(): #extracts the data from url
        pass
    
    def strip(): # Data preprocessing
        pass
    def pca(comp1, comp2): # Principal component analysis
        scaled = preprocessing.scale(self._data)
        pca = PCA()
        pca.fit(self._data)
        per_var = np.round(pca.explained_variance_ratio_*100, decimals= 1)
        pca_data = pca.transform(self-_data)
        labels = ["PC" + str(x) for x in range(1,len(per_var)+1)]
    def t_sne(): #t-distributed stochastic neighbor embedding
        pass
    def show():
        pass
class Plotter(): # not sure about that one yet
    pass