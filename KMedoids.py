#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:37:26 2022

@author: lerouxde
"""

from sklearn import metrics
import kmedoids
import time 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.io import arff

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
    
path = './artificial/'
databrut = arff.loadarff(open(path+"xclara.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

scores = []

for k in range (2, 25, 1):
    labels = kmedoidsFun(datanp, f0, f1, k)
    #scores.append(kmedoids.silhouette(euclidean_distances(datanp),labels)[0])
    scores.append(metrics.silhouette_score(euclidean_distances(datanp),labels))

print(scores)
plt.plot(range(2,25,1), scores, color = 'red', linestyle = 'dashed', linewidth = 2,
  markerfacecolor = 'blue', markersize = 5)
plt.title('Scores')
