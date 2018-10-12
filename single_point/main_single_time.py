#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 00:41:01 2018

@author: davitisoselia
"""
from . import merger


modeln='single_point/fall_detection_1.h5' # model name
merged_path = 'single_point/merged.csv'
import os.path

if not os.path.isfile(merged_path):
    merger.merge()

falls=[] #saves fall start-end moments

with open(merged_path) as csv:
    content = csv.readlines()
for i in range(len(content)):
    if('tart' in content[i]):
        falls.append([i])
    if('nd' in content[i]):
        falls[-1].append(i)
    content[i] = content[i].split(',')


#content = content[::10]

'''

    if (len(content[i][-1]) > 2):
        print(i)
   ''' 
import numpy as np
import sys





def row_to_numpy(point):
    segment = []
    fell = [0]
    if (int(content[point][-2])) > 0:
        fell = [1]
    segment = (content[point][:-2])
    for j in range(len(segment)):
        segment[j] = float(segment[j])
    segment = np.array(segment)
    return segment, fell

ml,mk = row_to_numpy(5)


sensorNum = ml.shape[0]



print(len((content[35232])))
print(len(content[0]))







from keras.models import Sequential
from keras.layers import LSTM
#from keras.layers import CuDNNLSTM as LSTM

from keras.layers import Dense
from keras.layers import Conv1D
import numpy as np
from keras.models import load_model

'''
if not os.path.isfile(modeln):
    model = Sequential()
    model.add(Dense(sensorNum, input_dim=183, activation='relu'))
    model.add(Dense(sensorNum/2, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model = load_model(modeln)
'''
model = Sequential()
model.add(Dense(183, input_dim=183, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import random

def get_fall(point = 0):
    fell = [0]
    while fell == [0]:
        point = falls[random.randint(0, len(falls))][0] + random.randint(10, 100)
        segment , fell = row_to_numpy(point)
    return segment , fell


from keras.models import load_model
j = 0
iter = 0
balance_needed = False
lastnp = np.array([])
temp_storage = '43.39	11.03	16.21	85.62	178.8	172.6	40.39	18.69	7.767	137.4	105.2	107.9	59.82	107.9	51.32	206.8	54.73	56.91	34.09	29	46.25	49.62	47.7	96.14	83.78	178.1	178.8	87.09	82.72	177.8	178.8	136.2	135.6	18.03	37.27	39.53	28.23	35.61	37.22	3.644	67.37	21.58	63.19	82.38	82.69	178.9	82.62	180	180	86.57	180	179.7	80.72	179.9	80.44	79.99	107.7	179.8	89.15	178.1	-14.97	66.98	180	50.26	71.05	109.1	179.9	84.46	179.8	180	86.76	180	179.8	89.57	179.8	180	86.59	179.4	179.8	86.43	179.9	179.8	87.1	179.8	179.7	88.29	179.7	1791	1377	2042	1881	1604	958.5	2965	3014	2036	2325	2396	2078	3940	3616	3647	4682	3080	1474	3861	6624	2780	4152	4875	5995	3818	2253	641.1	3324	4992	5953	2714	3502	1516	8552	2113	1574	4586	15890	14370	4300	2691	1406	14470	10660	2981	11250	16000	14240	0.9126	0.4787	0.7286	0.8671	0.01697	0.8622	0.7929	-0.03522	0.3384	0.7069	0.6081	0.8366	0.7314	0.6626	0.7502	0.6727	0.6694	0.9436	0.2001	0.8859	0.04322	0.5211	0.9018	0.9656	0.9665	0.2635	0.9052	0.9012	0.7843	0.7929	0.4846	0.8235	0.7375	0.7812	0.6371	0.7238	0.4928	0.4771	0.8543	0.7993	0.8932	0.6572	0.4866	0.01178	0.1309	0.718	0.7586	0.7251'.split('	')
normalizer = [] 
for value in temp_storage:
    normalizer.append(float(value))
temp_storage = np.array(normalizer)





confusion_matrix = [[0,0],[0,0]]
def checkresult_confusion(point = random.randint(1, len(content)-50), length = random.randint(300, 1500), check_fall = False, confusion_matrix = [[0,0],[0,0]]):
    np_arr, y = get_fall() if check_fall else row_to_numpy(point)
    np_arr = np_arr / temp_storage
    y_train = np.array(y)
    x_train = np.transpose(np_arr).reshape(1,sensorNum)
    prediction = model.predict(x_train)
    #print(y_train)
    #print(prediction)
    if (y_train[0]==round(prediction[0][0]) and y_train[0] == 1):
        confusion_matrix[0][0] += 1
    elif (y_train[0]==round(prediction[0][0]) and y_train[0] == 0):
        confusion_matrix[1][1] += 1
    elif (y_train[0]!=round(prediction[0][0]) and y_train[0] == 0):
        confusion_matrix[1][0] += 1
    elif (y_train[0]!=round(prediction[0][0]) and y_train[0] == 1):
        confusion_matrix[0][1] += 1
    return (y_train[0]==round(prediction[0][0])), confusion_matrix

#modeln='a.h5'
#model = load_model('a.h5')

def test():
    matrix = [[0,0],[0,0]]
    fall = True    
    correct = 0
    i = 0
    while i < 100:
        try:
            temp, matrix = checkresult_confusion(check_fall = fall, confusion_matrix =matrix )
            correct += (temp)            
            i+=1
            fall = not fall
        except:
            print(sys.exc_info()[0])
    
    print('accuracy: ')
    print(correct)
    print(matrix)
    
# train NN
while(iter<5000000):
    j=random.randint(1, len(content)-50)
    #avred = not avred
    try:
        #print(iter)
        print('Balance 0 : ' + str(balance_needed))
        if balance_needed:
            np_arr, y = get_fall()
        else:
            np_arr, y = row_to_numpy(j)
        print('Balance : ' + str(balance_needed))
        lastnp = np_arr
        np_arr = np_arr / temp_storage
        #x_train = x_train / 50
        y_train = np.array(y)
        x_train = np.transpose(np_arr).reshape(1,sensorNum)
        print('fit : ')
        model.fit(x_train, y_train)
        #print(j)
        #j=random.randint(1, 5)
        #j=random.randint(1264, 1896)
        if(iter % 1000 == 0):
            model.save(modeln)
            #test()
            print(iter)
        iter+=1;
        balance_needed = not balance_needed
        #print('here')
        print(iter)
    except (TypeError,IndexError):
        print('error raised at index ' +str(j))
        print(sys.exc_info()[0])
        pass
    except:
        print(sys.exc_info()[0])
        raise
        

    
'''
confusion_matrix = [[0,0],[0,0]]
def checkresult_confusion(point = random.randint(1, len(content)-50), length = random.randint(300, 1500), check_fall = False):
    np_arr, y = get_fall() if check_fall else generate_numpy(point, length)
    np_arr = np_arr / temp_storage
    y_train = np.array(y)
    x_train = np.transpose(np_arr).reshape(1,sensorNum)
    prediction = model.predict(x_train)
    print(y_train)
    print(prediction)
    if (y_train[0]==round(prediction[0][0]) and y_train[0] == 1):
        confusion_matrix[0][0] += 1
    elif (y_train[0]==round(prediction[0][0]) and y_train[0] == 0):
        confusion_matrix[1][1] += 1
    elif (y_train[0]!=round(prediction[0][0]) and y_train[0] == 0):
        confusion_matrix[1][0] += 1
    elif (y_train[0]!=round(prediction[0][0]) and y_train[0] == 1):
        confusion_matrix[0][1] += 1
    return (y_train[0]==round(prediction[0][0]))
'''
    
    
    
    
    
    
    
    
    
#train boosted decision tree
        
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

def random_forests_create():
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    return rf


def random_forests_train(rf, X_train, Y_train):
    rf.fit(X_train, Y_train)
    return rf
    
    
    #rf.fit(get_fall[0], get_fall[1])
    
    

