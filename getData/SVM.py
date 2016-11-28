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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.datasets import fetch_20newsgroups
from MushroomData import MushroomData
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def main():
    # print("testing data class")
    


    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    
    print('missing elements')

    # target = y_test.target
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)


    y_true = np.array(y_test)
    print "macro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='macro'))+ "\n"
    print "micro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='micro'))+ "\n"
    print "weighted precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='weighted'))+ "\n"

    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))


    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    
    print('\nAll Elements')

    # target = y_test.target
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    
    y_true = np.array(y_test)
    print "macro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='macro'))+ "\n"
    print "micro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='micro'))+ "\n"
    print "weighted precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='weighted'))+ "\n"
 
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
    
    pass

if __name__ == "__main__":
    main()
    