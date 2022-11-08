#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:33:26 2022

@author: lerouxde
"""
import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np

# datasets qui marchent bien :
# 2d-4c.arff 4 clusters distance_threshold = 10
# dartboard1.arff 4 clusters distance_threshold = 0.05
# smile3.arff 4 clusters distance_threshold = 0.05

# datasets qui marchent pas bien :
# square4.arff 4 clusters
# zelnik4.arff 5 ou 4 avec bruit

# path = './artificial/'
# name="banana"
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
    

# # set distance_threshold ( 0 ensures we compute the full tree )
# tps1 = time.time ()
# model = cluster.AgglomerativeClustering( distance_threshold = 15 , linkage = 'single' , n_clusters = None )
# model = model.fit( datanp )
# tps2 = time.time ()
# labels = model.labels_
# k = model.n_clusters_
# leaves = model.n_leaves_

# # Affichage clustering
# plt.scatter( f0 , f1 , c = labels , s = 8 )
# plt.title( " Resultat du clustering " )
# plt.show()
# print( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000,2 ) ," ms " )

# set the number of clusters
k = 5
tps1 = time.time()
model = cluster.AgglomerativeClustering( linkage = 'ward' , n_clusters = k )
model = model.fit( datanp )
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_

# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.savefig(name)
plt.show()
print( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000,2 ) ," ms " )