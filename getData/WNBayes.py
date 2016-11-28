#!/usr/bin/python

#WNBayes.py
#
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
#
#Purpose: A naive bayes classifier for the mushroom dataset 
#           with weighted attributes
#

import numpy as np
from MushroomData import MushroomData
from sklearn import tree
from sklearn import metrics


data = MushroomData()

y_test,X_test,y_train,X_train = data.get_datasets()

edible_probs = [{} for x in range(22)]
inedible_probs = [{} for x in range(22)]
feat_counts = data.feat_counts()


def fit(X, y):
    e_count = 0
    total = len(X)
    for i in range(total):
        if y[i] is 1:
            #edible
            e_count += 1.0
        for j in range(len(X[i])):
            if y[i] is 1:
                #edible
                if X[i][j] in edible_probs[j]:
                    edible_probs[j][X[i][j]] += 1.0
                else:
                    edible_probs[j][X[i][j]] = 1.0
            else:
                #inedible
                if X[i][j] in inedible_probs[j]:
                    inedible_probs[j][X[i][j]] += 1.0
                else:
                    inedible_probs[j][X[i][j]] = 1.0
                
    
    for i in range(len(edible_probs)):
        for j in range(1, feat_counts[i]+1):
            if j in edible_probs[i]:
                edible_probs[i][j] += 1
                edible_probs[i][j] /= (e_count + feat_counts[i])
            else:
                edible_probs[i][j] = 1.0 / e_count + feat_counts[i]
                
    for i in range(len(inedible_probs)):
        for j in range(1, feat_counts[i]+1):
            if j in inedible_probs[i]:
                inedible_probs[i][j] += 1
                inedible_probs[i][j] /= (e_count + feat_counts[i])
            else:
                inedible_probs[i][j] = 1.0 / e_count + feat_counts[i]
    
                
    
def predict(X,w):
    y = np.zeros(len(X))
    for i in range(len(X)):
        p_edible = 0
        p_inedible = 0
        for j in range(len(X[i])):
            if X[i][j] in edible_probs[j]:
                p_edible += w[j] * np.log(edible_probs[j][X[i][j]])
            if X[i][j] in inedible_probs[j]:
                p_inedible += w[j] * np.log(inedible_probs[j][X[i][j]])
        
        if p_edible > p_inedible:
            y[i] = 1
        else:
            y[i] = -1
    return y
            

fit(X_train, y_train)

print '*** Missing records eliminated***'
print '*** Equal weights ***'
weights = [1.0 for i in range(22)]
y_pred = predict(X_test, weights)
print 'accuracy = %f' %( np.mean((list(y_test)-y_pred)==0))
print(metrics.classification_report(y_test, y_pred, target_names=data.class_labels, digits=6))


print '*** Weight using decision tree depth = 3***'
print 'Tree depth 3'
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train,y_train)

y_pred = predict(X_test, clf.feature_importances_)
print 'accuracy = %f' %( np.mean((list(y_test)-y_pred)==0))
print(metrics.classification_report(y_test, y_pred, target_names=data.class_labels, digits=6))


print '*** Missing record not eliminated***'
y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
fit(X_train, y_train)

print '*** Equal weights ***'
weights = [1.0 for i in range(22)]
y_pred = predict(X_test, weights)
print 'accuracy = %f' %( np.mean((list(y_test)-y_pred)==0))
print(metrics.classification_report(y_test, y_pred, target_names=data.class_labels, digits=6))


print '*** Weight using decision tree depth = 3***'
print 'Tree depth 3'
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train,y_train)

y_pred = predict(X_test, clf.feature_importances_)
print 'accuracy = %f' %( np.mean((list(y_test)-y_pred)==0))
print(metrics.classification_report(y_test, y_pred, target_names=data.class_labels, digits=6))


print '*** Missing attribute ignored***'
y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False, ignore=['stalk-root'])
fit(X_train, y_train)

print '*** Equal weights ***'
weights = [1.0 for i in range(22)]
y_pred = predict(X_test, weights)
print 'accuracy = %f' %( np.mean((list(y_test)-y_pred)==0))
print(metrics.classification_report(y_test, y_pred, target_names=data.class_labels, digits=6))


print '*** Weight using decision tree depth = 3***'
print 'Tree depth 3'
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train,y_train)

y_pred = predict(X_test, clf.feature_importances_)
print 'accuracy = %f' %( np.mean((list(y_test)-y_pred)==0))
print(metrics.classification_report(y_test, y_pred, target_names=data.class_labels, digits=6))

