# -*- coding: utf-8 -*-
print('hello')
import os
os.chdir('RNN')
os.listdir('/')
import main
import test
os.chdir('../')
os.chdir('single_point')
import main_single_time.py
os.chdir('../')
os.chdir('bdt')
import bdt
import train_der
