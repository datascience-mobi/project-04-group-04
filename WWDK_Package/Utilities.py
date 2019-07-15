import sklearn.cluster as sk
from WWDK_Package import Data as d
from WWDK_Package import Cluster as cl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import time

def time_k_plot(data, iterations, runs):
    liste = [0]
    sklearn_liste = [0]
    #inet_liste = []
    for i in range(iterations):
        meantime = []
        sk_meantime = []
        for j in range(runs):
            
            
            start = time.time()
            lib = cl.Kmeans(inits=10, method="rng", k=i+1)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
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
            
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
        #inet_liste.append(np.mean(inet_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste, label='WWDK')
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r", label='sklearn')
    plt.plot(sklearn_liste, "kx")
    #plt.plot(inet_liste, "g")
    #plt.plot(inet_liste, "kx")
    plt.xlabel("k")
    plt.ylabel("time[s]")
    plt.legend()
    return plt.show()

def inertia_k_plot(data, iterations, runs):
    inertia = [0]
    sklearn_inertia = [0]
    #inet_liste = []
    for i in range(iterations):
        meaninertia = []
        sk_meaninertia = []
        for j in range(runs):

            lib = cl.Kmeans(inits=10, method="rng", k=i+1)
            lib.fit(data)
            meaninertia.append(lib.inertia_)
            
            lib = sk.KMeans(init="random", n_init=10, n_clusters=i+1)
            lib.fit(data)
            sk_meaninertia.append(lib.inertia_)

        inertia.append(np.mean(meaninertia))
        sklearn_inertia.append(np.mean(sk_meaninertia))
        #inet_liste.append(np.mean(inet_meantime)
        
    plt.plot(inertia, label='WWDK')
    plt.plot(inertia, "kx")
    plt.plot(sklearn_inertia, "r", label='sklearn')
    plt.plot(sklearn_inertia, "kx")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.legend()
    return plt.show()

def time_init_plot(data, iterations, runs):
    liste = [0]
    sklearn_liste = [0]
    
    for i in range(iterations):
        meantime =[]
        sk_meantime = []
        for j in range(runs):
            
            
            start = time.time()
            lib = cl.Kmeans(inits=i+1, method="rng", k=8)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
            
            
            start = time.time()
            lib = sk.KMeans(init="random",n_clusters=8, n_init=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantime.append(end-start)
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste, label='WWDK')
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r", label='sklearn')
    plt.plot(sklearn_liste, "kx")
    plt.xlabel("inits")
    plt.ylabel("time[s]")
    plt.legend()
    return plt.show()

def elbow_plot(data, max_k):
    Sum_of_squared_distances = []
    for i in range(max_k):
        km = cl.Kmeans(inits=10, method="rng", k=i+1)
        km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(Sum_of_squared_distances, "kx")
    plt.plot(Sum_of_squared_distances)
    plt.xlabel("k")
    plt.ylabel("Sum of squared distances")
    return plt.show()

def plot(data, k):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(k):
        graph = pd.DataFrame(data._data[np.argwhere(data.labels_ == i)].squeeze())
        center = pd.DataFrame(data.cluster_centers_[i]).T
        #print("Cluster"+ str(i) +  " -- Assigned Points \n" + str(graph))
        ax.plot(graph[0], graph[1], "o")
        ax.plot(center[0],center[1], "kx")
        ax.annotate("Cluster " + str(i), xy = (center[0],center[1])
            )
    return plt.show()

def plot_Compare(data, dist, clusters,k):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(k):
        graph = pd.DataFrame(data[np.argwhere(dist == i)].squeeze())
        center = pd.DataFrame(clusters[i]).T
        #print("Cluster"+ str(i) +  " -- Assigned Points \n" + str(graph))
        ax.plot(graph[0], graph[1], "o")
        ax.plot(center[0],center[1], "kx")
        ax.annotate("Cluster " + str(i), xy = (center[0],center[1])
            )   
    return plt.show()