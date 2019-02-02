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
needed = ["shank", "foot", 'hip', 'wrist']
for sensor_name in (needed): 
    X_train = generate_numpys.X_t
    Y_train = generate_numpys.Y_t
    X_test = generate_numpys.X_test
    Y_test = generate_numpys.Y_test
    X_train = selected_features.select_features(X_train, sensor_name)
    X_test = selected_features.select_features(X_test, sensor_name)
    
    model = Sequential()
    model.add(DropoutNoScale(drop_in, input_shape=(X_train.shape[1],), name='drop0'))
    for i in range(num_hidden):
        model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
                  name='dense{}'.format(i+1)))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
        model.add(Activation(binary_tanh, name='act{}'.format(i+1)))
        model.add(DropoutNoScale(drop_hidden, name='drop{}'.format(i+1)))
    model.add(BinaryDense(1, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
              name='dense'))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))
    
    
    
    opt = Adam(lr=lr_start) 
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    
    y_pred = score = model.predict(X_test)
    
    
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
    with open('bnn_fpr_list_1.pkl', 'wb') as f:
        pickle.dump(fpr_list, f)
    with open('bnn_tpr_list_1.pkl', 'wb') as f:
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
plt.title('BNN')
plt.legend(loc="lower right")
plt.savefig('bnn.png')







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