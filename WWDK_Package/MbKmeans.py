
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
import numba
from numba import njit, jit
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

class MiniBatchKMeans:
    """Performs mini-batch k-means on a set of input data."""
    def __init__(self, k=8, inits=300, max_iterations=300, tol=1e-3, batch_size=128):
        """Simple mini-batch k-means clustering implementation in pure Python.
        
        Args:
            k (int): number of clusters to fit.
            inits (int): number of independent initializations to perform.
            max_iterations (int): maximum number of iterations to perform.
            tol (float): tolerance for early stopping.
        """
        self._k = k
        self._inits = inits
        self._max_iterations = max_iterations
        self._tol = tol
        self._batch_size = batch_size
        
    def create_batch(self, data): #wählt ein random batch aus den Daten aus
        data_batch = np.random.choice(range(len(data)), self._batch_size, replace=False)
        return data[data_batch]
        
    def initialize(self, data): #setzt zufällig centroids aus den Daten
        indices = np.random.choice(range(len(data)), self._k, replace=False)
        return data[indices], np.zeros(self._k)
    
    def expectation(self, data, centroids): #ordnet den centroids die Punkte zu
        centroids = np.expand_dims(centroids, axis=1)
        data = np.expand_dims(data, axis=0)
        metric = np.linalg.norm(centroids - data, axis=2)
        return np.argmin(metric, axis=0)
    
    @staticmethod
    @numba.jit(nopython=True)
    def _maximization_aux(data, assignments, centroids, centroid_count): #verschiebt die centroids richtung clustermittelpunkt mit lr
        update = centroids.copy()
        for idx, assignment in enumerate(assignments):
            data_point = data[idx]
            centroid_count[assignment] += 1
            lr = 1 / centroid_count[assignment] #learning rate
            update[assignment] = update[assignment] * (1 - lr) + data_point * lr
        return update
    
    def maximization(self, data, assignments, centroids, centroid_count): 
        return MiniBatchKMeans._maximization_aux(data, assignments, centroids, centroid_count)
    
    def final_assignments(self, data, centroids): 
        assignments = []
        for idx in range(len(data) // self._batch_size + 1):
            start = idx * self._batch_size
            stop = min(idx * self._batch_size + self._batch_size, len(data))
            sub_result = self.expectation(data[start:stop], centroids)
            assignments.append(sub_result)
        return np.concatenate(assignments, axis=0)
    
    def fit(self, data): #alles zusammen
        centroids, counts = self.initialize(data)
        
        old_centroids = None
        for idx in range(self._max_iterations):
            old_centroids = centroids.copy()
            
            batch = self.create_batch(data)
            assignments = self.expectation(batch, centroids)
            centroids = self.maximization(batch, assignments, centroids, counts)
            
            if np.linalg.norm(centroids - old_centroids) < self._tol:
                break

        result = self.final_assignments(data, centroids)
                
        return centroids, result
