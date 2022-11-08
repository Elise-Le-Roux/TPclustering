#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:54:44 2022

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
databrut = arff.loadarff(open(path+"2d-4c.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
scores1 = []
scores2 = []
scores3 = []

# set distance_threshold ( 0 ensures we compute the full tree )

for linkage in ["ward", "complete", "average", "single"] :
    scores1 = []
    scores2 = []
    scores3 = []
    for distance_threshold in np.arange(0.5, 20.0, 0.5) :
        
            tps1 = time.time ()
            model = cluster.AgglomerativeClustering( distance_threshold = distance_threshold , linkage = linkage , n_clusters = None )
            model = model.fit( datanp )
            tps2 = time.time ()
            labels = model.labels_
            k = model.n_clusters_
            leaves = model.n_leaves_
            try:
                scores1.append(metrics.silhouette_score(datanp,labels))
                scores2.append(metrics.davies_bouldin_score(datanp,labels))
                scores3.append(metrics.calinski_harabasz_score(datanp,labels))
            except:
                pass

    plt.scatter(np.arange(0.5, 0.5 * len(scores1)+0.5, 0.5), scores1, color = 'red')
    plt.title('Silhouette score')
    plt.show()
    
    plt.scatter(np.arange(0.5, 0.5 * len(scores2)+0.5, 0.5), scores2, color = 'blue')
    plt.title('Davies Bouldin score')
    plt.show()
    
    plt.scatter(np.arange(0.5, 0.5 * len(scores3)+0.5, 0.5), scores3, color = 'green')
    plt.title('Calinski Harabasz score')
    plt.show()
    
# plt.plot(range(2,len(scores2) + 2, 1), scores2, color = 'blue', linestyle = 'dashed', linewidth = 2,
#   markerfacecolor = 'blue', markersize = 5)


# for i in range(len(scores3)):
#     scores3[i]= scores3[i]/10000
    
# plt.plot(range(2,len(scores3) + 2, 1), scores3, color = 'green', linestyle = 'dashed', linewidth = 2,
#   markerfacecolor = 'blue', markersize = 5)