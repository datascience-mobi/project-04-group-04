import sklearn.cluster as sk
from WWDK_Package import Data as d
from WWDK_Package import Cluster as cl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import time

def time_k_plot(data, iterations, runs):
    liste = []
    sklearn_liste = []
    #inet_liste = []
    for i in range(iterations):
        for j in range(runs):
            
            meantime =[]
            start = time.time()
            lib = cl.Kmeans(inits=10, method="rng", k=i+1, tol= 1)
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
            sk_meantime = []
            start = time.time()
            lib = sk.KMeans(init="random",n_init=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantime.append(end-start)
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
        #inet_liste.append(np.mean(inet_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste)
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r")
    plt.plot(sklearn_liste, "kx")
    #plt.plot(inet_liste, "g")
    #plt.plot(inet_liste, "kx")
    plt.xlabel("k")
    plt.ylabel("time[s]")
    return plt.show()

def time_init_plot(data, iterations, runs):
    liste = []
    sklearn_liste = []
    
    for i in range(iterations):
        for j in range(runs):
            
            meantime =[]
            start = time.time()
            lib = cl.Kmeans(inits=i+1, method="rng", k=8,tol= 1)
            lib.fit(data)
            end = time.time()
            meantime.append(end-start)
            
            sk_meantime = []
            start = time.time()
            lib = sk.KMeans(init="random",n_init=i+1)
            lib.fit(data)
            end = time.time()
            sk_meantime.append(end-start)
        liste.append(np.mean(meantime))
        sklearn_liste.append(np.mean(sk_meantime))
     
    #print(lib.inertia_)
    plt.plot(liste)
    plt.plot(liste, "kx")
    plt.plot(sklearn_liste, "r")
    plt.plot(sklearn_liste, "kx")
    plt.xlabel("inits")
    plt.ylabel("time[s]")
    return plt.show()

def elbow_plot(data, iterations):
    liste = []
    for i in range(iterations):
        lib = cl.Kmeans(inits=10, method="rng", k=i+1)
        lib.fit(data)
        liste.append(lib.inertia_)
        #print(lib.inertia_)
    plt.plot(liste, "kx")
    plt.plot(liste)
    plt.xlabel("k")
    plt.ylabel("score")
    return plt.show()

def plot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(data._k):
        graph = pd.DataFrame(data._data[np.argwhere(data.labels_ == i)].squeeze())
        center = pd.DataFrame(data.cluster_centers_[i]).T
        #print("Cluster"+ str(i) +  " -- Assigned Points \n" + str(graph))
        ax.plot(graph[0], graph[1], "o")
        ax.plot(center[0],center[1], "kx")
        ax.annotate("Cluster " + str(i), xy = (center[0],center[1])
            )
    return plt.show()