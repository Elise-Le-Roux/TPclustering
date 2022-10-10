#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:16:42 2022

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

import numpy as np
import matplotlib.pyplot as plt
import time 
from sklearn import cluster
from sklearn import metrics
from scipy.io import arff

path = './artificial/'
databrut = arff.loadarff(open(path+"smile1.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

print ("Appel KMeans pour une valeur fixee de k")
tps1 = time.time()
k=4
model = cluster.KMeans(n_clusters=k , init='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter(f0 , f1 , c=labels , s=8)
plt.title ( " Donnees apres clustering Kmeans " )
plt.show()
print("nb clusters =" ,k , " , nb iter =", iteration , 
      " , . . . . . . runtime = " , round((tps2 - tps1)*1000,2), " ms " )

score1 = metrics.silhouette_score(datanp,labels)
score2 = metrics.davies_bouldin_score(datanp,labels)

print(score2)