#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:11:18 2022

@author: lerouxde
"""


# datasets qui marchent bien :
# twodiamonds.arff k=2
# xclara.arff k=2
# 2d-4c.arff k=4
# curves1.arff k=2
# diamond9.arff k=9
# sizes1.arff k=4
# spherical_4_3.arff k=4
# twenty.arff k=20

# datasets qui ne marchent pas bien :
# dense-disk-3000.arff k=2
# long1.arff k=2
# smile1.arff k=4

import numpy as np
import matplotlib.pyplot as plt
import time 
from sklearn import cluster
from sklearn import metrics
from scipy.io import arff


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
databrut = arff.loadarff(open(path+"dense-disk-3000.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
scores1 = []
scores2 = []
scores3 = []

for k in range (2, 25, 1):
    labels = kmeans(datanp, f0, f1, k)
    scores1.append(metrics.silhouette_score(datanp,labels))
    scores2.append(metrics.davies_bouldin_score(datanp,labels))
    scores3.append(metrics.calinski_harabasz_score(datanp,labels))

plt.plot(range(2,25,1), scores1, color = 'red', linestyle = 'dashed', linewidth = 2,
  markerfacecolor = 'blue', markersize = 5)
plt.title('Scores')

plt.plot(range(2,25,1), scores2, color = 'blue', linestyle = 'dashed', linewidth = 2,
  markerfacecolor = 'blue', markersize = 5)


for i in range(23):
    scores3[i]= scores3[i]/10000
    
plt.plot(range(2,25,1), scores3, color = 'green', linestyle = 'dashed', linewidth = 2,
  markerfacecolor = 'blue', markersize = 5)