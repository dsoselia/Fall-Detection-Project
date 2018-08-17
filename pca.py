# -*- coding: utf-8 -*-


import numpy as np
import sklearn
from sklearn.decomposition import PCA
a = np.array([[2,2, 0],[2,2, 0], [1,1, 0]])


a = np.array([[2,2, 0],[2,2, 0], [1,2, 0]])

a = np.array([[2,2, 0],[2,2, 0], [50,50, 0]])

a = np.array([[1,2,3,4], [1,2,3,4] , [1,2,3,4] , [1,2,3,4]])

a = np.array([[1,2,3,4], [2,1,3,4] , [3,0,3,4] , [4,-1,3,4]])

a = np.array([[4, 6, 10],[ 3, 10, 13],[-2, -6, -8]])


pca = PCA(n_components=3)
pca.fit(a)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)

from sklearn.covariance import EmpiricalCovariance

E = EmpiricalCovariance()
E.fit(a)
print(E.covariance_)
from numpy import linalg as LA
w, v = LA.eig(E.covariance_)