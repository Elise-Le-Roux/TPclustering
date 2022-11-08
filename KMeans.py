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

# path = './artificial/'
# name="spiral"
# databrut = arff.loadarff(open(path+name+".arff", 'r'))
# datanp = [[x[0], x[1]] for x in databrut[0]]
# f0 = [f[0] for f in datanp]
# f1 = [f[1] for f in datanp]

path ='./dataset-rapport/'
name="zz2"
databrut = np.loadtxt(path+name+".txt", unpack = True)
f0 = databrut[0]
f1 = databrut[1]

datanp=[]
for x in range(len(f0)) :
    datanp.append([f0[x], f1[x]])
    

print ("Appel KMeans pour une valeur fixee de k et données : " + name)
tps1 = time.time()
k=5
model = cluster.KMeans(n_clusters=k , init='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter(f0 , f1 , c=labels, s=8,) #cmap="hsv"
#plt.title ( " Donnees apres clustering Kmeans " )
plt.savefig(name)
plt.show()
print("nb clusters =" ,k , " , nb iter =", iteration , 
      " , . . . . . . runtime = " , round((tps2 - tps1)*1000,2), " ms " )



score1 = metrics.silhouette_score(datanp,labels)
score2 = metrics.davies_bouldin_score(datanp,labels)
score3 = metrics.calinski_harabasz_score(datanp,labels)

print("Silhouette_score =", score1)
print("Davies_bouldin_score =", score1)
print("Calinski_harabasz_score =", score1)