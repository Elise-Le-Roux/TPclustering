#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:54:26 2022

@author: lerouxde
"""

import hdbscan
import time
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import metrics
import numpy as np
from sklearn import neighbors


# path = './artificial/'
# name="compound"
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

tps1 = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(datanp)
tps2 = time.time()

n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

# Affichage clustering
plt.scatter( f0 , f1 , c = cluster_labels , s = 8 )
plt.title( " Resultat du clustering " )
print(  " nb clusters = "  , n_clusters_," runtime = " , round (( tps2 - tps1 ) * 1000,2 ) ," ms " )
plt.savefig(name)

plt.show()