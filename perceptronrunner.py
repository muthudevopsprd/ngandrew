#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:07:03 2020

@author: msvmuthu
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Perceptron as ptrn
"""
s = os.path.join('https://archive.ics.uci.edu', 'ml',
... 'machine-learning-databases',
... 'iris','iris.data')
"""
df=pd.read_csv('iris.data',header=None,encoding='utf-8')
print (df.tail())
"""
      0    1    2    3               4
145  6.7  3.0  5.2  2.3  Iris-virginica
146  6.3  2.5  5.0  1.9  Iris-virginica
147  6.5  3.0  5.2  2.0  Iris-virginica
148  6.2  3.4  5.4  2.3  Iris-virginica
149  5.9  3.0  5.1  1.8  Iris-virginic

0. sepal length in cm 
1. sepal width in cm 
2. petal length in cm 
3. petal width in cm 
4. class: 
-- Iris Setosa 1-50
-- Iris Versicolour 50-100 
-- Iris Virginica 100-150

"""

y=df.iloc[0:100,4].values

y=np.where(y== 'Iris-setosa',-1,1)
X=df.iloc[0:100, [0,2]].values

plt.scatter(X[:50,0],X[:50,1],color='red', marker='+',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue', marker='x',label='versicolor')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc='upper left')
plt.show()

ppn=ptrn.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_)+1),ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
print ('done')