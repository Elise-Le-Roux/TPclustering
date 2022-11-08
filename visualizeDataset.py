#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:06:09 2022

@author: lerouxde
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff

# path = './artificial/'
# databrut = arff.loadarff(open(path+"3MC.arff", 'r'))
# data = [[x[0], x[1]] for x in databrut[0]]

# f0 = [f[0] for f in data]
# f1 = [f[1] for f in data]

path ='./dataset-rapport/'
name="y1"
databrut = np.loadtxt(path+name+".txt", unpack = True)
f0 = databrut[0]
f1 = databrut[1]

datanp=[]
for x in range(len(f0)) :
    datanp.append([f0[x], f1[x]])
    
    
f0, f1 = np.loadtxt(path+name+".txt", usecols =(0, 1), unpack = True)

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()