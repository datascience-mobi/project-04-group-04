'''Cluster contains all of the different cluster methods.'''
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import numba
from numba import njit, jit
from statistics import mean
from sklearn.datasets.samples_generator import make_blobs
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class Kmeans(BaseEstimator, ClusterMixin, TransformerMixin):               
    """Performs native k-means on a set of input data.  """
    def __init__(self, inits=10, k=8, maxit=300, method="++", tol=1e-3):
        """Simple k-means clustering implementation in pure Python.
        
        Kmeans implements the :class:`BaseEstimator`, :class:`ClusterMixin`
        and :class:`TransformerMixin` interfaces of Sci-kit learn.
        It can be used as a drop-in replacement for the ``sklearn`` implementation.

        Args:
            k (int): number of clusters to fit.
            inits (int): number of independent initializations to perform.
            max_iterations (int): maximum number of iterations to perform.
            method (str): method of choosing starting centers "++" or "rng"
            tol (float): tolerance for early stopping.

        Example::
        
            >>> data = np.random.randn((100, 2))
            >>> kmeans = Kmeans(k=3)
            >>> result = kmeans.fit(data)
        """
        
        self.labels_ = None
        self.cluster_centers_ = None
        self.inits = inits
        self.k = k
        self.maxit = maxit
        self.method = method
        self.tol = tol

    def fit(self, data):
        """Fits cluster centers to data and labels data accordingly.
        
        Args:
            data (:class:`np.array`): [SIZE, DIMENSION] array of input data.

        Returns:
            A fit estimator containing centroids and labels on data.
        """
        self._data = data
        best_clust = float('inf')
        
        for i in (range(self.inits)):
            # random points from the dataset are selected as starting centers
            if self.method == "rng": # random centers are choosen
                dot = np.random.choice(range(len(self._data)), self.k, replace=False)
                self.cluster_centers_ = self._data[dot]
            elif self.method == "++": # kmeans++ is initiated
                clusters = np.zeros((self.k, self._data.shape[1]))
                dot = np.random.choice(len(self._data), replace=False) # one random center
                clusters[0] = self._data[dot]
                exp_clusters = np.expand_dims(clusters, axis=1)
                exp_data = np.expand_dims(self._data, axis=0) # clusters and data are expanded to be easily substracted in a next step
                for i in range (self.k - 1): #the rest of the centers are chosen based on the first one
                    D = np.min(np.sum(np.square(exp_clusters[0:i + 1] - exp_data), axis=2), axis=0)
                    r = np.random.random()
                    ind = np.argwhere(np.cumsum(D / np.sum(D)) >= r)[0][0] # the point when the cummulative sum is equal to r is choosen as ind
                    clusters[i + 1] = self._data[ind]
                self.cluster_centers_ = clusters
            else:
                raise AttributeError("No valid method") # if a non existing method is choosen, a error is raised

            old_centroids = None

            for i in range(self.maxit):
                old_centroids = self.cluster_centers_.copy() # the cluster centers are copied for tolerance later on.
                clusters = np.expand_dims(self.cluster_centers_, axis=1)
                data = np.expand_dims(self._data, axis=0)
                eucl = np.linalg.norm(clusters - data, axis=2) # euclidean dist by using integrated numpy function
                self.labels_ = np.argmin(eucl, axis = 0)
                for i in range(self.k): # range of clusters
                    position = np.where(self.labels_ == i) # position of points assosiated with cluster i are calculated
                    
                    if np.any(self.labels_ == i) == False:
                        self.cluster_centers_[i] = self._data[np.random.choice(self._data.shape[0], 1, replace=False)] # in rare events it can happen that no points are assigned to a cluster. if that happens centroid is newly choosen
                    else:
                        self.cluster_centers_[i] = self._data[position].mean(axis=0)
                overall_quality = np.sum(np.min(eucl.T ** 2, axis=1)) #quality of the clustering based on the inner cluster distances
                if overall_quality < best_clust:
                    best_clust = overall_quality
                    best_dist = self.labels_
                    best_centers = self.cluster_centers_
                if np.linalg.norm(self.cluster_centers_ - old_centroids) < self.tol: #if the tolerance is reached the algorithm is stopped. this enables faster running times
                    break
            self.cluster_centers_ = best_centers #final centers
            self.labels_ = best_dist #final labels
            self.inertia_ = best_clust # best quality reached in all attempts
                
        return self

    def predict(self, X):
        """Predicts cluster labels for a given dataset."""
        clusters = np.expand_dims(self.cluster_centers_, axis=1)
        data = np.expand_dims(X, axis=0)
        eucl = np.linalg.norm(clusters-data, axis=2) # euclidean dist by using integrated numpy function
        self.labels_ = np.argmin(eucl, axis = 0)
        return self.labels_ 

    def transform(self, X):
        """Creates a matrix with distance to centroid for each point."""
        clusters = np.expand_dims(self.cluster_centers_, axis=1)
        data = np.expand_dims(X, axis=0)
        eucl = np.linalg.norm(clusters-data, axis=2)
        return eucl.T 

class MiniBatchKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Performs mini-batch k-means on a set of input data."""
    def __init__(self, k=8, inits=300, max_iterations=300, tol=1e-3, batch_size=100, method = "++"):
        """Simple mini-batch k-means clustering implementation in pure Python.
        
        Args:
            k (int): number of clusters to fit.
            inits (int): number of independent initializations to perform.
            max_iterations (int): maximum number of iterations to perform.
            tol (float): tolerance for early stopping.
            batch_size (int): number of datapoints for the minibatch.
            method (str): method of choosing starting centers "++" or "rng"
        """
        self.labels_ = None
        self.cluster_centers_ = None
        self.k = k
        self.inits = inits
        self.max_iterations = max_iterations
        self.tol = tol
        self.batch_size = batch_size
        self.method = method
        
    def create_batch(self, data):
        """chooses x (x = batch_size) random points from the data to create the data batch"""
        self._data = data
        data_batch = np.random.choice(range(len(data)), self.batch_size, replace=False)
        return data[data_batch]
        
    def initialize(self, data):
        """chooses k random data points from data, to set centers for clustering."""
        if self.method == "rng":
            indices = np.random.choice(range(len(data)), self.k, replace=False)
            return data[indices], np.zeros(self.k)
        elif self.method == "++":
            clusters = np.zeros((self.k,2))
            dot = np.random.choice(len(data), replace=False) # random startpunkt
            clusters[0] = data[dot]
            exp_clusters = np.expand_dims(clusters, axis=1)
            exp_data = np.expand_dims(data, axis=0)
            for i in range (self.k-1):
                D = np.min(np.sum(np.square(exp_clusters[0:i+1]-exp_data),axis=2),axis=0)
                r = np.random.random()
                ind = np.argwhere(np.cumsum(D/np.sum(D)) >= r)[0][0]
                clusters[i+1] = data[ind]
            self.cluster_centers_ = clusters
            return clusters, np.zeros(self.k)
        else:
            raise AttributeError("No valid method")
    
    def expectation(self, data, centroids): 
        """measures the euclidean distance between each data_batch points
        and center points using numpys linalg.norm function"""
        centroids = np.expand_dims(centroids, axis=1)
        data = np.expand_dims(data, axis=0)
        metric = np.linalg.norm(centroids - data, axis=2)
        return np.argmin(metric, axis=0)

    @staticmethod
    @numba.jit(nopython=True)
    def _maximization_aux(data, assignments, centroids, centroid_count):
        """Moves the centroids to the new centroid point of the assigned data_batch points.
        But not completely, but according to the learning rate"""
        update = centroids.copy()
        for idx, assignment in enumerate(assignments):
            data_point = data[idx]
            centroid_count[assignment] += 1
            lr = 1 / centroid_count[assignment] # learning rate
            update[assignment] = update[assignment] * (1 - lr) + data_point * lr
        return update

    def maximization(self, data, assignments, centroids, centroid_count):
        """This part applies maximization_aux on the data using maximization_aux"""
        return MiniBatchKMeans._maximization_aux(data, assignments, centroids, centroid_count)
    
    def final_assignments(self, data, centroids):
        """Assignes the rest of the data points to the centroids,
        which were determined before (not only batch_points)"""
        assignments = []
        for idx in range(len(data) // self.batch_size + 1):
            start = idx * self.batch_size
            stop = min(idx * self.batch_size + self.batch_size, len(data))
            sub_result = self.expectation(data[start:stop], centroids)
            assignments.append(sub_result)
        return np.concatenate(assignments, axis=0)
    
    def fit(self, data):
        centroids, counts = self.initialize(data)
        
        old_centroids = None
        for idx in range(self.max_iterations):
            old_centroids = centroids.copy()
            
            batch = self.create_batch(data)
            assignments = self.expectation(batch, centroids)
            centroids = self.maximization(batch, assignments, centroids, counts)
            
            if np.linalg.norm(centroids - old_centroids) < self.tol:
                break

        self.labels_ = self.final_assignments(data, centroids)
        self.cluster_centers_ = centroids
                
        return self
    
    def predict(self, data):
        centroids = np.expand_dims(self.cluster_centers_, axis=1)
        data = np.expand_dims(data, axis=0)
        metric = np.linalg.norm(centroids - data, axis=2)
        self.labels_ = np.argmin(metric, axis=0)

        return self.labels_
    
    def transform(self, data):
        centroids = np.expand_dims(self.cluster_centers_, axis=1)
        data = np.expand_dims(data, axis=0)
        metric = np.linalg.norm(centroids - data, axis=2)

        return metric.T
