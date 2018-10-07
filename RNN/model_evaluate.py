# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 19:06:23 2018

@author: Dati
"""
import numpy as np
def percision(matrix):
    matrix = np.array(matrix)
    return matrix[0,0]/((matrix[0,0]+matrix[0,1]))

def recall(matrix):
    matrix = np.array(matrix)
    return matrix[0,0]/((matrix[0,0]+matrix[1,0]))

def specificity(matrix):
    matrix = np.array(matrix)
    return matrix[1,1]/((matrix[1,1]+matrix[0,1]))