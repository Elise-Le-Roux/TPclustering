#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:29:40 2022

@author: lerouxde
"""

import scipy.cluster.hierarchy as shc
from scipy.io import arff
import matplotlib.pyplot as plt

path = './artificial/'
databrut = arff.loadarff(open(path+"xclara.arff", 'r'))
datanp = [[x[0], x[1]] for x in databrut[0]]

# Donnees dans datanp
print( " Dendrogramme ’ single ’ donnees initiales " )
linked_mat = shc.linkage ( datanp , 'single')
plt.figure ( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat,
                orientation = 'top',
                distance_sort = 'descending',
                show_leaf_counts = False )
plt.show ()