#!/usr/bin/python

# GaussianNB.py
#
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
#
#
import numpy as np
import sys
import os
from sklearn.naive_bayes import GaussianNB
from MushroomData import MushroomData
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def main():
    data = MushroomData()

    print('\nGaussian Naive Bayes: (eliminating missing elements)')
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    
    print('\nGaussian Naive Bayes: (using all elements)')
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

    print('\nGaussian Naive Bayes: (Ignore stalk-root)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False, ignore=['stalk-root'])

    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

if __name__ == "__main__":
    main()
     
