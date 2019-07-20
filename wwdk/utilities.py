"""contains utility functions, that are used for comparison
of the package against sklearn. Also some function to visualize kmeans."""
import sklearn.cluster as sk
from wwdk import data as d
from wwdk import cluster as cl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import pandas as pd
import time
import math
import seaborn as sns 
import imageio
import shutil
import os
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

def time_k_plot(data, iterations, runs):
    liste = [0]
    listeplus = [0]
    sklearn_liste = [0]
    sklearn_listeplus = [0]
    #inet_liste = []
    for i in range(iterations):
        meantime = []
        sk_meantime = []
        meantimeplus = []
        sk_meantimeplus = []
        
        for j in range(runs):
            
            
            start = time.time()
            lib = cl.Kmeans(inits=10, method="rng", k=i+1)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
            
            start = time.time()
            lib = cl.Kmeans(inits=10, k=i+1)
            lib.fit(data)
            end = time.time()
            meantimeplus.append(end-start)
            '''
            inet_meantime =[]
            start = time.time()
            k_means(data,i+1,300)
            end = time.time()
            inet_meantime.append(end-start)
            '''
            start = time.time()
            lib = sk.KMeans(init="random", n_init=10, n_clusters=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantime.append(end-start)
            
            start = time.time()
            lib = sk.KMeans(n_init=10, n_clusters=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantimeplus.append(end-start)
            
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
        listeplus.append(np.mean(meantimeplus))
        sklearn_listeplus.append(np.mean(sk_meantimeplus))
        #inet_liste.append(np.mean(inet_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste, label='WWDK')
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r", label='sk')
    plt.plot(sklearn_liste, "kx")
    plt.plot(listeplus, label='WWDK_++', linestyle='dashed')
    plt.plot(listeplus, "kx")
    plt.plot(sklearn_listeplus, "r", label='sk_++', linestyle='dashed')
    plt.plot(sklearn_listeplus, "kx")
    #plt.plot(inet_liste, "g")
    #plt.plot(inet_liste, "kx")
    plt.xlabel("k")
    plt.ylabel("time[s]")
    plt.legend()
    return plt.show()

def time_k_plot_mb(data, iterations, runs):  
    liste = [0]
    listeplus = [0]
    sklearn_liste = [0]
    sklearn_listeplus = [0]
    #inet_liste = []
    for i in range(iterations):
        meantime = []
        sk_meantime = []
        meantimeplus = []
        sk_meantimeplus = []
        
        for j in range(runs):
            
            
            start = time.time()
            lib = cl.MiniBatchKMeans(method="rng", k=i+1)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
            
            start = time.time()
            lib = cl.MiniBatchKMeans(k=i+1)
            lib.fit(data)
            end = time.time()
            meantimeplus.append(end-start)
      
            start = time.time()
            lib = sk.MiniBatchKMeans(init="random", n_clusters=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantime.append(end-start)
            
            start = time.time()
            lib = sk.MiniBatchKMeans(n_clusters=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantimeplus.append(end-start)
            
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
        listeplus.append(np.mean(meantimeplus))
        sklearn_listeplus.append(np.mean(sk_meantimeplus))
        #inet_liste.append(np.mean(inet_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste, label='WWDK_mb')
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r", label='sk_mb')
    plt.plot(sklearn_liste, "kx")
    plt.plot(listeplus, label='WWDK_mb_++', linestyle='dashed')
    plt.plot(listeplus, "kx")
    plt.plot(sklearn_listeplus, "r", label='sk_mb_++', linestyle='dashed')
    plt.plot(sklearn_listeplus, "kx")
    #plt.plot(inet_liste, "g")
    #plt.plot(inet_liste, "kx")
    plt.xlabel("k")
    plt.ylabel("time[s]")
    plt.legend()
    return plt.show()

def time_k_wwdk_compare (data, iterations, runs, batchsize=100): #compare our vanilla, ++ and mb 
    liste = [0]
    listeplus = [0]
    listemb = [0]
    listembplus = [0]
    #inet_liste = []
    for i in range(iterations):
        meantime = []
        meantimemb = []
        meantimeplus = []
        meantimembplus = []
        
        for j in range(runs):
            
            
            start = time.time()
            lib = cl.Kmeans(method="rng", k=i+1)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
            
            start = time.time()
            lib = cl.Kmeans(k=i+1)
            lib.fit(data)
            end = time.time()
            meantimeplus.append(end-start)
            
            start = time.time()
            lib = cl.MiniBatchKMeans(method="rng", k=i+1, batch_size=batchsize)
            lib.fit(data)
            end = time.time()
            meantimemb.append(end-start)
            
            start = time.time()
            lib = cl.MiniBatchKMeans(k=i+1, batch_size=batchsize)
            lib.fit(data)
            end = time.time()
            meantimembplus.append(end-start)
            
        liste.append(np.mean(meantime))
        listeplus.append(np.mean(meantimeplus))
        listemb.append(np.mean(meantimemb))
        listembplus.append(np.mean(meantimembplus))
        #inet_liste.append(np.mean(inet_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste, label='WWDK')
    plt.plot(liste, "kx")
    plt.plot(listeplus, label='WWDK_++', linestyle='dashed')
    plt.plot(listeplus, "kx")
    plt.plot(listemb, label='WWDK_mb')
    plt.plot(listemb, "kx")
    plt.plot(listembplus, label='WWDK_mb_++', linestyle='dashed')
    plt.plot(listembplus, "kx")
    #plt.plot(inet_liste, "g")
    #plt.plot(inet_liste, "kx")
    plt.xlabel("k")
    plt.ylabel("time[s]")
    plt.legend()
    return plt.show()

def inertia_k_plot(data, iterations, runs):
    inertia = [float("nan")]
    sklearn_inertia = [float("nan")]
    inertiaplus = [float("nan")]
    sklearn_inertiaplus = [float("nan")]
    #inet_liste = []
    for i in range(iterations):
        meaninertia = []
        sk_meaninertia = []
        meaninertiaplus = []
        sk_meaninertiaplus = []
        for j in range(runs):

            lib = cl.Kmeans(inits=10, method="rng", k=i+1)
            lib.fit(data)
            meaninertia.append(lib.inertia_)
            
            lib = cl.Kmeans(inits=10, k=i+1)
            lib.fit(data)
            meaninertiaplus.append(lib.inertia_)
            
            lib = sk.KMeans(init="random", n_init=10, n_clusters=i+1)
            lib.fit(data)
            sk_meaninertia.append(lib.inertia_)
            
            lib = sk.KMeans(n_init=10, n_clusters=i+1)
            lib.fit(data)
            sk_meaninertiaplus.append(lib.inertia_)

        inertia.append(np.mean(meaninertia))
        sklearn_inertia.append(np.mean(sk_meaninertia))
        inertiaplus.append(np.mean(meaninertiaplus))
        sklearn_inertiaplus.append(np.mean(sk_meaninertiaplus))
        #inet_liste.append(np.mean(inet_meantime)
        
    plt.plot(inertia, label='WWDK')
    plt.plot(inertia, "kx")
    plt.plot(sklearn_inertia, "r", label='sk')
    plt.plot(sklearn_inertia, "kx")
    plt.plot(inertiaplus, label='WWDK_++', linestyle='dashed')
    plt.plot(inertiaplus, "kx")
    plt.plot(sklearn_inertiaplus, "r", label='sk_++', linestyle='dashed')
    plt.plot(sklearn_inertiaplus, "kx")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.legend()
    return plt.show()

def time_init_plot(data, iterations, runs):
    liste = [0]
    sklearn_liste = [0]
    listeplus = [0]
    sklearn_listeplus = [0]
    
    for i in range(iterations):
        meantime =[]
        sk_meantime = []
        meantimeplus =[]
        sk_meantimeplus = []
        for j in range(runs):
            
            
            start = time.time()
            lib = cl.Kmeans(inits=i+1, method="rng", k=8)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
            
            start = time.time()
            lib = cl.Kmeans(inits=i+1, k=8)
            lib.fit(data)
            end = time.time()
            meantimeplus.append(end-start)
            
            
            start = time.time()
            lib = sk.KMeans(init="random",n_clusters=8, n_init=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantime.append(end-start)
            
            start = time.time()
            lib = sk.KMeans(n_clusters=8, n_init=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantimeplus.append(end-start)
            
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
        listeplus.append(np.mean(meantimeplus))
        sklearn_listeplus.append(np.mean(sk_meantimeplus))
     
    #print(lib.inertia_)
    plt.plot(liste, label='WWDK')
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r", label='sk')
    plt.plot(sklearn_liste, "kx")
    plt.plot(listeplus, label='WWDK_++', linestyle='dashed')
    plt.plot(listeplus, "kx")
    plt.plot(sklearn_listeplus, "r", label='sk_++', linestyle='dashed')
    plt.plot(sklearn_listeplus, "kx")
    plt.xlabel("inits")
    plt.ylabel("time[s]")
    plt.legend()
    return plt.show()

def elbow_plot(data, max_k):
    Sum_of_squared_distances = [float("nan")]
    for i in range(max_k):
        km = cl.Kmeans(inits=10, method="rng", k=i+1)
        km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(Sum_of_squared_distances, "kx")
    plt.plot(Sum_of_squared_distances)
    plt.xlabel("k")
    plt.ylabel("Sum of squared distances")
    return plt.show()


def plot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(data.k):
        graph = pd.DataFrame(data._data[np.argwhere(data.labels_ == i)].squeeze())
        center = pd.DataFrame(data.cluster_centers_[i]).T
        #print("Cluster"+ str(i) +  " -- Assigned Points \n" + str(graph))
        ax.plot(graph[0], graph[1], "o", markersize=1)
        ax.plot(center[0],center[1], "o", c="k", markersize=3.5)
        ax.annotate("  Cluster " + str(i), xy = (center[0],center[1]))
            
    return plt.show()

def plot_seaborn(data, ks = 8, methods = "rng"):
    
    van_umap = cl.Kmeans(inits = 10, method = methods, k = ks).fit(data)
    
    y_van_umap = van_umap.predict(data)
    
    centers_van_umap = van_umap.cluster_centers_

    sns.set_style("dark")
    g = sns.scatterplot(x= data[:, 0], y= data[:, 1], hue= y_van_umap, s=10, palette="hot")
    sns.scatterplot(x=centers_van_umap[:, 0], y=centers_van_umap[:, 1], s=50)
    
    plt.show()    

def plot_compare(data, dist, clusters,k, title="title"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(k):
        graph = pd.DataFrame(data[np.argwhere(dist == i)].squeeze())
        center = pd.DataFrame(clusters[i]).T
        ax.plot(graph[0], graph[1], "o")
        ax.plot(center[0],center[1], "kx")
        ax.annotate("Cluster " + str(i), xy = (center[0],center[1])
            )
        black_patch = mpatches.Patch(color='k', label=title) 
        plt.legend(handles=[black_patch])
    return plt.show()


class Gifcreator(BaseEstimator, ClusterMixin, TransformerMixin):               
    """Performs native k-means on a set of input data """
    def __init__(self, inits=10, k=8, maxit=300, method="++", tol = 1e-3):
 
        self.labels_ = None
        self.cluster_centers_ = None
        self.inits = inits
        self.k = k
        self.maxit = maxit
        self.method = method
        self.tol = tol
       
    """fits given data and calculates cluster centers and labels points accordingly"""

    def create_gif(self,data):
        os.mkdir("./plots")
        filenames = []
        self.data = data
        best_clust = float('inf')
        
        for c in (range(self.inits)):
            print("Init: " + str(c))
            """random points from the dataset are selected as starting centers """
            if self.method == "rng": # random centers are choosen
                
                dot = np.random.choice(self.data.shape[0], self.k, replace=False)
                self.cluster_centers_ = self.data[dot]
            elif self.method == "++": # kmeans++ is initiated
                clusters = np.zeros((self.k, self.data.shape[1]))
                dot = np.random.choice(len(self.data), replace=False) # one random center
                clusters[0] = self.data[dot]
                exp_clusters = np.expand_dims(clusters, axis=1)
                exp_data = np.expand_dims(self.data, axis=0) # clusters and data are expanded to be easily substracted in a next step
                for i in range (self.k - 1): #the rest of the centers are chosen based on the first one
                    D = np.min(np.sum(np.square(exp_clusters[0:i + 1] - exp_data), axis=2), axis=0)
                    r = np.random.random()
                    ind = np.argwhere(np.cumsum(D / np.sum(D)) >= r)[0][0] # the point when the cummulative sum is equal to r is choosen as ind
                    clusters[i + 1] = self.data[ind]
                self.cluster_centers_ = clusters
            else:
                raise AttributeError("No valid method")

            old_centroids = None

            for i in range(self.maxit):
                #plt.scatter(X[:, 0], X[:, 1],c="w", s=50)
                for ie in range(self.k):
                    graph = pd.DataFrame(self.data[np.argwhere(self.labels_ == ie)].squeeze())
                    center = pd.DataFrame(self.cluster_centers_[ie]).T
                    plt.plot(graph[0], graph[1], "o")
                    plt.plot(center[0],center[1], "kx")
               
                #print(i)
                plt.savefig("./plots/graph" +str(c+1)+"-"+ str(i+1)+ ".png")
                filenames.append("./plots/graph" +str(c+1)+"-"+ str(i+1)+ ".png")
                plt.clf()
                
                
                old_centroids = self.cluster_centers_.copy()
                clusters = np.expand_dims(self.cluster_centers_, axis=1)
                data = np.expand_dims(self.data, axis=0)
                eucl = np.linalg.norm(clusters-data, axis=2) # euclidean dist by using integrated numpy function
                self.labels_ = np.argmin(eucl, axis=0)
                
               # print(i)
                #print(self.cluster_centers_)
                #print(self.labels_)
                #if i > 30:
                  #  error = True
                 #   break
                for i in range(self.k): # range of clusters
                    position = np.where(self.labels_ == i)
                    # Position of points assosiated with cluster i are calculated.
                    
                    if np.any(self.labels_ == i) == False:
                        self.cluster_centers_[i] = self.data[np.random.choice(self.data.shape[0], 1, replace=False)]
                        # In rare events it can happen that no points are assigned to a cluster. 
                        # If that happens centroid is newly choosen
                    else:
                        self.cluster_centers_[i] = self.data[position].mean(axis=0)
                    #out = pd.DataFrame(data[np.argwhere(dist == i)].squeeze())
                overall_quality = np.sum(np.min(eucl.T, axis=1))
                if overall_quality < best_clust:
                    best_clust = overall_quality
                    best_dist = self.labels_
                    best_centers = self.cluster_centers_
                if np.linalg.norm(self.cluster_centers_ - old_centroids) < self.tol:
                    break
            self.cluster_centers_ = best_centers
            self.labels_ = best_dist
            self.inertia_ = best_clust
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
            imageio.mimsave('./kmeans.gif', images)
                
        shutil.rmtree("./plots")
        print("Gif created!")
            
        return self 

