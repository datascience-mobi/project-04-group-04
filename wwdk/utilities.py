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
    inertia = [0]
    sklearn_inertia = [0]
    inertiaplus = [0]
    sklearn_inertiaplus = [0]
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
    Sum_of_squared_distances = []
    for i in range(max_k+1):
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
        ax.plot(graph[0], graph[1], "o")
        ax.plot(center[0],center[1], "kx")
        ax.annotate("Cluster " + str(i), xy = (center[0],center[1])
            )
    return plt.show()

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