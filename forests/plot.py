# -*- coding: utf-8 -*-
from . import selected_features

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import keras.backend as K
import pickle
from bnn.binary_ops import binary_tanh as binary_tanh_op
from bnn.binary_layers import BinaryDense
from sklearn.ensemble import RandomForestClassifier

class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    '''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

def binary_tanh(x):
    return binary_tanh_op(x)

batch_size = 100
nb_epoch = 100
nb_classes = 10

H = 'Glorot'
kernel_lr_multiplier = 'Glorot'

# network
num_unit = 120
num_hidden = 1
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / nb_epoch)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
drop_in = 0.2
drop_hidden = 0.5




lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
from . import generate_numpys
print("merged ...")






'''
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
'''
fpr_list = []
tpr_list = []
needed = ['hip lt',"shank lt", "foot lt", 'wrist lt', "shank rt", "foot rt", 'hip rt', 'wrist rt']
for sensor_name in (needed): 
    X_train = generate_numpys.X_t
    Y_train = generate_numpys.Y_t
    X_test = generate_numpys.X_test
    Y_test = generate_numpys.Y_test
    X_train = selected_features.select_features(X_train, sensor_name)
    X_test = selected_features.select_features(X_test, sensor_name)
    

    

    

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
    
    
    rf = random_forests_create()
    rf = random_forests_train(rf,X_train , Y_train)
    y_pred = score = rf.predict(X_test)
    log_y = Y_test
    log_predicted = y_pred
    
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    #log_y = np.array([1, 1, 0, 0])
    #log_predicted =  np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(log_y, log_predicted, pos_label  = 1)
    roc_auc = auc(log_y, log_predicted, reorder  = True)
    fpr_list.append(fpr)
    tpr_list.append(tpr)    
    with open('forests_fpr_list_1.pkl', 'wb') as f:
        pickle.dump(fpr_list, f)
    with open('forests_tpr_list_1.pkl', 'wb') as f:
        pickle.dump(tpr_list, f)

    # Plot ROC curve
    #plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    
for graph in range(len(needed)):
    plt.plot(fpr_list[graph], tpr_list[graph], label= needed[graph])
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specificity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('RF')
plt.legend(loc="lower right")
plt.savefig('forests.png')







def get_labels(path = 'labels.csv'):
    with open(path) as csv:
        content = csv.readlines()[0].lower()
        content  = content.replace('","', "TARA")
        content  = content.replace(',', " ")
        content  = content.replace('"', " ")
        content  = content.replace('\n', " ")
        content  = content.replace('TARA', " , ")
        content = content.split(",")
        return content
    
def get_indexes (term = 'shank'):
    labels = get_labels()
    return [b for b in range(len(labels)) if term in labels[b]]