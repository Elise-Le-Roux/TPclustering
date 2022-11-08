#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:11:37 2022

@author: lerouxde
"""

from sklearn import metrics
import kmedoids
import time 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.io import arff
from sklearn import cluster

def kmedoidsFun(data, f0, f1, k):
    tps1 = time.time()
    distmatrix = euclidean_distances(data)
    fp = kmedoids.fasterpam(distmatrix, k)
    tps2 = time.time()
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    print( " Loss with FasterPAM : " , fp.loss)
    plt.scatter(f0 , f1 , c=labels_kmed , s =8)
    plt.title( " Donnees apres clustering KMedoids " )
    plt.show()
    print( " nb clusters =" ,k , " , nb iter =" , iter_kmed , " , . . . . . . runtime = " , 
          round((tps2 - tps1) * 1000,2 ) , "ms")
    return labels_kmed

def kmeans(data, f0, f1, k) :
    print ("Appel KMeans pour une valeur fixee de k")
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k , init='k-means++')
    model.fit(data)
    tps2 = time.time()
    labels = model.labels_
    iteration = model.n_iter_
    plt.scatter(f0 , f1 , c=labels , s=8)
    plt.title ( " Donnees apres clustering Kmeans " )
    plt.show()
    print("nb clusters =" ,k , " , nb iter =", iteration , 
          " , . . . . . . runtime = " , round((tps2 - tps1)*1000,2), " ms " )
    return labels
    
    
path = './artificial/'
databrut = arff.loadarff(open(path+"spiral.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

k=9
labelsKMedoids = kmedoidsFun(datanp, f0, f1, k)
labelsKMeans = kmeans(datanp, f0, f1, k)

print(metrics.rand_score(labelsKMedoids, labelsKMeans))
print(metrics.mutual_info_score(labelsKMedoids, labelsKMeans))