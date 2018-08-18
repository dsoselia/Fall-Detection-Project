# -*- coding: utf-8 -*-

labels = '"Cervical Flexion,deg","Cervical Lateral - RT,deg","Cervical Axial - RT,deg","Lumbar Flexion,deg","Lumbar Lateral - RT,deg","Lumbar Axial - RT,deg","Thoracic Flexion,deg","Thoracic Lateral - RT,deg","Thoracic Axial - RT,deg","Elbow Flexion LT,deg","Elbow Flexion RT,deg","Shoulder Total Flexion LT,deg","Shoulder Total Flexion RT,deg","Shoulder Flexion LT,deg","Shoulder Flexion RT,deg","Shoulder Abduction LT,deg","Shoulder Abduction RT,deg","Shoulder Rotation - out LT,deg","Shoulder Rotation - out RT,deg","Wrist Extension LT,deg","Wrist Extension RT,deg","Wrist Radial LT,deg","Wrist Radial RT,deg","Wrist Supination LT,deg","Wrist Supination RT,deg","Hip Flexion LT,deg","Hip Flexion RT,deg","Hip Abduction LT,deg","Hip Abduction RT,deg","Hip Rotation - out LT,deg","Hip Rotation - out RT,deg","Knee Flexion LT,deg","Knee Flexion RT,deg","Ankle Dorsiflexion LT,deg","Ankle Dorsiflexion RT,deg","Ankle Inversion LT,deg","Ankle Inversion RT,deg","Ankle Abduction LT,deg","Ankle Abduction RT,deg","Head course,deg","Head pitch,deg","Head roll,deg","Upper spine course,deg","Upper spine pitch,deg","Upper spine roll,deg","Upper arm course LT,deg","Upper arm pitch LT,deg","Upper arm roll LT,deg","Forearm course LT,deg","Forearm pitch LT,deg","Forearm roll LT,deg","Hand course LT,deg","Hand pitch LT,deg","Hand roll LT,deg","Upper arm course RT,deg","Upper arm pitch RT,deg","Upper arm roll RT,deg","Forearm course RT,deg","Forearm pitch RT,deg","Forearm roll RT,deg","Hand course RT,deg","Hand pitch RT,deg","Hand roll RT,deg","Lower spine course,deg","Lower spine pitch,deg","Lower spine roll,deg","Pelvis course,deg","Pelvis pitch,deg","Pelvis roll,deg","Thigh course LT,deg","Thigh pitch LT,deg","Thigh roll LT,deg","Shank course LT,deg","Shank pitch LT,deg","Shank roll LT,deg","Foot course LT,deg","Foot pitch LT,deg","Foot roll LT,deg","Thigh course RT,deg","Thigh pitch RT,deg","Thigh roll RT,deg","Shank course RT,deg","Shank pitch RT,deg","Shank roll RT,deg","Foot course RT,deg","Foot pitch RT,deg","Foot roll RT,deg","Head Accel Sensor X,mG","Head Accel Sensor Y,mG","Head Accel Sensor Z,mG","Upper spine Accel Sensor X,mG","Upper spine Accel Sensor Y,mG","Upper spine Accel Sensor Z,mG","Upper arm Accel Sensor X LT,mG","Upper arm Accel Sensor Y LT,mG","Upper arm Accel Sensor Z LT,mG","Forearm Accel Sensor X LT,mG","Forearm Accel Sensor Y LT,mG","Forearm Accel Sensor Z LT,mG","Hand Accel Sensor X LT,mG","Hand Accel Sensor Y LT,mG","Hand Accel Sensor Z LT,mG","Upper arm Accel Sensor X RT,mG","Upper arm Accel Sensor Y RT,mG","Upper arm Accel Sensor Z RT,mG","Forearm Accel Sensor X RT,mG","Forearm Accel Sensor Y RT,mG","Forearm Accel Sensor Z RT,mG","Hand Accel Sensor X RT,mG","Hand Accel Sensor Y RT,mG","Hand Accel Sensor Z RT,mG","Lower spine Accel Sensor X,mG","Lower spine Accel Sensor Y,mG","Lower spine Accel Sensor Z,mG","Pelvis Accel Sensor X,mG","Pelvis Accel Sensor Y,mG","Pelvis Accel Sensor Z,mG","Thigh Accel Sensor X LT,mG","Thigh Accel Sensor Y LT,mG","Thigh Accel Sensor Z LT,mG","Shank Accel Sensor X LT,mG","Shank Accel Sensor Y LT,mG","Shank Accel Sensor Z LT,mG","Foot Accel Sensor X LT,mG","Foot Accel Sensor Y LT,mG","Foot Accel Sensor Z LT,mG","Thigh Accel Sensor X RT,mG","Thigh Accel Sensor Y RT,mG","Thigh Accel Sensor Z RT,mG","Shank Accel Sensor X RT,mG","Shank Accel Sensor Y RT,mG","Shank Accel Sensor Z RT,mG","Foot Accel Sensor X RT,mG","Foot Accel Sensor Y RT,mG","Foot Accel Sensor Z RT,mG","Head Rot X,","Head Rot Y,","Head Rot Z,","Upper spine Rot X,","Upper spine Rot Y,","Upper spine Rot Z,","LT Upper arm Rot X,","LT Upper arm Rot Y,","LT Upper arm Rot Z,","LT Forearm Rot X,","LT Forearm Rot Y,","LT Forearm Rot Z,","LT Hand Rot X,","LT Hand Rot Y,","LT Hand Rot Z,","RT Upper arm Rot X,","RT Upper arm Rot Y,","RT Upper arm Rot Z,","RT Forearm Rot X,","RT Forearm Rot Y,","RT Forearm Rot Z,","RT Hand Rot X,","RT Hand Rot Y,","RT Hand Rot Z,","Lower spine Rot X,","Lower spine Rot Y,","Lower spine Rot Z,","Pelvis Rot X,","Pelvis Rot Y,","Pelvis Rot Z,","LT Thigh Rot X,","LT Thigh Rot Y,","LT Thigh Rot Z,","LT Shank Rot X,","LT Shank Rot Y,","LT Shank Rot Z,","LT Foot Rot X,","LT Foot Rot Y,","LT Foot Rot Z,","RT Thigh Rot X,","RT Thigh Rot Y,","RT Thigh Rot Z,","RT Shank Rot X,","RT Shank Rot Y,","RT Shank Rot Z,","RT Foot Rot X,","RT Foot Rot Y,","RT Foot Rot Z,"'
labels = labels.split('","')
len(labels)

import numpy as np
import sklearn
from sklearn.decomposition import PCA
a = np.array([[2,2, 0],[2,2, 0], [1,1, 0]])


a = np.array([[2,2, 0],[2,2, 0], [1,2, 0]])

a = np.array([[2,2, 0],[2,2, 0], [50,50, 0]])

a = np.array([[1,2,3,4], [1,2,3,4] , [1,2,3,4] , [1,2,3,4]])

a = np.array([[1,2,3,4], [2,1,3,4] , [3,0,3,4] , [4,-1,3,4]])

a = np.array([[4, 6, 10],[ 3, 10, 13],[-2, -6, -8]])
from sklearn import preprocessing
a = X_t
a = preprocessing.normalize(a)
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


import matplotlib.pyplot as plt

X = a
y = Y_t
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

plt.figure()
colors = ['navy', 'turquoise']
target_names = ['navy', 'turquoise'] 
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')

plt.figure()

for i in range(50):
    plt.scatter(X_r[i, 0], X_r[i, 1], color=colors[y[i]])   

j = abs(E.covariance_[0])
indices = np.where(j > 0.9*j.max())
j = abs(pca.components_[0])
indices = np.where(j > 0.9*j.max())
    
    




accuracys_train = []
accuracys_test = []
for i in range(len(labels)):
    model = XGBClassifier()
    model.fit(X_t[:,i].reshape([X_t.shape[0],1]), Y_t)
    y_pred = model.predict(X_t[:,i].reshape([X_t.shape[0],1]))
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_t, predictions)
    #print("Training Accuracy: %.2f%%" % (accuracy * 100.0))
    #print(sklearn.metrics.precision_score(Y_t, predictions))
    #print(sklearn.metrics.recall_score(Y_t, predictions))
    
    
    accuracys_train.append(accuracy)
    y_pred = model.predict(X_test[:,i].reshape([X_test.shape[0],1]))
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_test, predictions)
    #print("Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    import sklearn
    #print(sklearn.metrics.precision_score(Y_test, predictions))
    #print(sklearn.metrics.recall_score(Y_test, predictions))
    accuracys_test.append(accuracy)

j = abs(np.array(accuracys_test)) 
indices = np.where(j > 0.9*j.max())

CSV = np.array([labels, pca.components_[0], pca.components_[1], pca.components_[2], accuracys_train, accuracys_test])
np.savetxt("PCA.csv", a, delimiter=",")
