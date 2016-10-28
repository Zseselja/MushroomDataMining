#!/usr/bin/python

#MushroomData.py
#
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
#
#Purpose: Get the mushroom data, transform it into a format usable by
#           the data mining algorithms, split into training and test sets,
#           and process results.
#
import numpy as np
import sys
import os
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from MushroomData import MushroomData


if __name__ == "__main__":
    print("testing data class")
    


    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    
    print('missing elements')

    # target = y_test.target
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))


    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    
    print('All Elements')

    # target = y_test.target
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)
 
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
