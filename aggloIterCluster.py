#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:54:43 2022

@author: lerouxde
"""

import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import metrics
import numpy as np

# datasets qui marchent bien :
# 2d-4c.arff 4 clusters distance_threshold = 10
# dartboard1.arff 4 clusters distance_threshold = 0.05
# smile3.arff 4 clusters distance_threshold = 0.05

path = './artificial/'
databrut = arff.loadarff(open(path+"smile3.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
scores1 = []
scores2 = []
scores3 = []


for linkage in ["ward", "complete", "average", "single"] :
    scores1 = []
    scores2 = []
    scores3 = []
    for k in range(2,10,1) :
            tps1 = time.time()
            model = cluster.AgglomerativeClustering( linkage = 'single' , n_clusters = k )
            model = model.fit( datanp )
            tps2 = time.time()
            labels = model.labels_
            kres = model.n_clusters_
            leaves = model.n_leaves_
            try:
                scores1.append(metrics.silhouette_score(datanp,labels))
                scores2.append(metrics.davies_bouldin_score(datanp,labels))
                scores3.append(metrics.calinski_harabasz_score(datanp,labels))
            except:
                pass

    plt.scatter(range(2,10,1), scores1, color = 'red')
    plt.title('Silhouette score')
    plt.show()
    
    plt.scatter(range(2,10,1), scores2, color = 'blue')
    plt.title('Davies Bouldin score')
    plt.show()
    
    plt.scatter(range(2,10,1), scores3, color = 'green')
    plt.title('Calinski Harabasz score')
    plt.show()