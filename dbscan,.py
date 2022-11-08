#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:07:54 2022

@author: lerouxde
"""

import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import metrics
import numpy as np
from sklearn import neighbors

# smile3.arff eps=0.03, min_samples= 4
# xclara.arff eps=eps=5, min_samples = 7

# 2d-4c-no4.arff
# aggregation.arff

# path = './artificial/'
# name="spiralsquare"
# databrut = arff.loadarff(open(path+name+".arff", 'r'))
# datanp = [[x[0], x[1]] for x in databrut[0]]
# f0 = [f[0] for f in datanp]
# f1 = [f[1] for f in datanp]

path ='./dataset-rapport/'
name="zz1"
databrut = np.loadtxt(path+name+".txt", unpack = True)
f0 = databrut[0]
f1 = databrut[1]

datanp=[]
for x in range(len(f0)) :
    datanp.append([f0[x], f1[x]])

tps1 = time.time()
model = cluster.DBSCAN(eps=10000, min_samples = 10)
model = model.fit( datanp )
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.savefig(name)
plt.show()
print( " nb clusters = "  , n_clusters_, " , nb feuilles = "  , " runtime = " , round (( tps2 - tps1 ) * 1000,2 ) ," ms " )


# Distances k plus proches voisins
# Donnees dans X
k = 10
neigh = neighbors.NearestNeighbors( n_neighbors=k )
neigh.fit (datanp )
distances , indices = neigh.kneighbors( datanp )
# retirer le point " origine "
newDistances = np.asarray ( [np.average(distances[i][1:] ) for i in range (0 ,distances.shape [0])])
trie = np.sort (newDistances)
plt.title ( " Plus proches voisins ( 3 ) " )
plt.plot ( trie ) ;
plt.show ()


# SCORES
#print("Silhouette score : ", metrics.silhouette_score(datanp,labels))
#print("Davies Bouldin score : ", metrics.davies_bouldin_score(datanp,labels))
#print("Calinski Harabasz score : ", metrics.calinski_harabasz_score(datanp,labels))