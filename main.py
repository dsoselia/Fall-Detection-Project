#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 00:41:01 2018

@author: davitisoselia
"""


falls=[] #saves fall start-end moments
with open('merged.csv') as csv:
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



def generate_numpy(point, length = 500):
    segment = []
    falls = 0;
    fell = [[0,1]]
    for i in range(point, point + length):
        if ('all' in content[i][-1]):
            falls+=1
        if i%10==0:
            segment.append(content[i][:-2])
    if (falls == 1):
        return
    elif(falls>1):
        fell = [[1,0]]
    for i in range(len(segment)):
        for j in range(len(segment[i])):
            segment[i][j] = float(segment[i][j])
    segment = np.array(segment)
    return segment, fell

ml,mk = generateNumpy(5)







sensorNum = ml.shape[1]



print(len((content[35232])))
print(len(content[0]))







from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
import numpy as np


model = Sequential()
model.add(LSTM(25, return_sequences=True, stateful=True, input_shape=(None, sensorNum),
         batch_input_shape=(1, None, sensorNum)))
model.add(LSTM(20, recurrent_dropout=0.2))
#model.add(Dense(30, activation='relu'))
#model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])




import random


def get_fall(point = 0, length = random.randint(300, 1500)):
    if point == 0:
        point = falls[random.randint(0, len(falls))][0] - random.randint(100, 500)
    segment , fell = generate_numpy(point, length)
    return segment , fell


j = 0
iter = 0
modeln='abcdeg_l.h5'
balance_needed = False
import sys

while(iter<50000):
    j=random.randint(1, len(content)-50)
    #avred = not avred
    try:
        #print(iter)
        if balance_needed:
            np_arr, y = get_fall()
        else:
            np_arr, y = generate_numpy(j)
        x_train = np.transpose(np_arr).reshape(1,np_arr.shape[0],np_arr.shape[1])
        x_train = x_train / 50
        y_train = np.array(y)
        model.fit(x_train, y_train, batch_size=1, nb_epoch=1, shuffle=False, verbose=0)
        #print(j)
        #j=random.randint(1, 5)
        #j=random.randint(1264, 1896)
        if(iter % 1000 == 0):
            model.save(modeln)
            print(iter)
        iter+=1;
        balance_needed = not balance_needed
    except (TypeError,IndexError):
        print('error raised at index ' +str(j))
    except:
        print(sys.exc_info()[0])
        raise