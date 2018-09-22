# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:02:00 2018

@author: Dati
"""


X = []
Y = []
iter = 0
# prep numpy for random forest
balance_needed = False
while(iter<len(content)):
    j=iter
    iter+=1
    if iter%(int(len(content)/100))==0:
        print(iter)
    #print(j)
    #avred = not avred
    try:
        #print(iter)
        #print('Balance 0 : ' + str(balance_needed))
        if balance_needed:
            np_arr, y = get_fall()
        else:
            np_arr, y = row_to_numpy(j)
        #print('Balance : ' + str(balance_needed))
        lastnp = np_arr
        np_arr = np_arr / temp_storage
        y_train = np.array(y)
        x_train = np.transpose(np_arr).reshape(sensorNum)
        X.append(x_train)
        Y.append(y_train)
        iter+=1;
        balance_needed = False
    except (TypeError,IndexError):
        print('error raised at index ' +str(j))
        print(sys.exc_info()[0])
        pass
    except:
        print(sys.exc_info()[0])
        raise



j = 0
iter = 0
balance_needed = False
lastnp = np.array([])

X_1 = []
Y_1 = []
iter = 0
# prep numpy for random forest
balance_needed = False
while(iter<400):
    j=random.randint(1, int((len(content)-50)))
    #print(j)
    #avred = not avred
    try:
        #print(iter)
        #print('Balance 0 : ' + str(balance_needed))
        if balance_needed:
            np_arr, y = get_fall()
        else:
            np_arr, y = row_to_numpy(j)
        #print('Balance : ' + str(balance_needed))
        lastnp = np_arr
        np_arr = np_arr / temp_storage
        y_train = np.array(y)
        x_train = np.transpose(np_arr).reshape(sensorNum)
        X_1.append(x_train)
        Y_1.append(y_train)
        iter+=1;
        balance_needed = not balance_needed
    except (TypeError,IndexError):
        print('error raised at index ' +str(j))
        print(sys.exc_info()[0])
        pass
    except:
        print(sys.exc_info()[0])
        raise

 
X_t = np.array(X)
Y_t = np.array(Y)

X = np.concatenate((X_t,nth_derivative(X_t)),1)

Y_t = Y_t.reshape(Y_t.shape[0])
X_test = np.array(X_1)
Y_test = np.array(Y_1)
Y_test = Y_test.reshape(Y_test.shape[0])


j = Y_t
index_fall = np.where(j== 1)[0]
index_nofall =  np.where(j== 0)[0]








X_train = []
Y_train = []
balance_needed = False
iter = 0
while(iter<2000):
    if balance_needed:
        j= index_fall[random.randint(1, index_fall.shape[0] )]
    else:
        j= index_nofall[random.randint(1, index_nofall.shape[0] )]
    #print('Balance : ' + str(balance_needed))
    X_train.append(Z[j])
    Y_train.append(Y_t[j])
    iter+=1;
    balance_needed = not balance_needed
X_train = np.array(X_train)
Y_train = np.array(Y_train)



X_test = []
Y_test = []
balance_needed = False
iter = 0
while(iter<400):
    if balance_needed:
        j= index_fall[random.randint(1, index_fall.shape[0]-50)]
    else:
        j= index_nofall[random.randint(1, index_nofall.shape[0]-50)]
    #print('Balance : ' + str(balance_needed))
    X_test.append(Z[j])
    Y_test.append(Y_t[j])
    iter+=1;
    balance_needed = not balance_needed
X_test = np.array(X_test)
Y_test = np.array(Y_test)

model = XGBClassifier()
model.fit(X_train, Y_train)







y_pred = model.predict(X_train)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_train, predictions)
print("Training Accuracy: %.2f%%" % (accuracy * 100.0))
import sklearn
print(sklearn.metrics.precision_score(Y_train, predictions))
print(sklearn.metrics.recall_score(Y_train, predictions))





y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Testing Accuracy: %.2f%%" % (accuracy * 100.0))
import sklearn
print(sklearn.metrics.precision_score(Y_test, predictions))
print(sklearn.metrics.recall_score(Y_test, predictions))

BDT_scores = model.feature_importances_
j = abs(np.array(BDT_scores)) 
indices = np.where(j > 0.3*j.max())

BDT_scores = model.feature_importances_
j = abs(np.array(BDT_scores)) 
indices1 = np.where(j < 0.3*j.max())



np.sum(model.feature_importances_[indices])/np.sum(model.feature_importances_)













accuracys_train = []
accuracys_test = []
for i in range(X.shape[1]):
    model = XGBClassifier()
    model.fit(X_train[:,i].reshape([X_train.shape[0],1]), Y_train)
    y_pred = model.predict(X_train[:,i].reshape([X_train.shape[0],1]))
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_train, predictions)
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
indices = np.where(j > 0.99*j.max()) 






real_accuracy= []       
for i in range(len(labels)):
    real_accuracy.append(accuracys_test[i]+accuracys_test[i+183])
j = abs(np.array(real_accuracy)) 
indices = np.where(j > 0.95*j.max())
 
CSV = np.array([accuracys_test, real_accuracy])
np.savetxt("realacc.csv",CSV, delimiter=",", fmt='%s')
