#!/usr/bin/python

# SVM.py
#
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics



def main():
    
    # SVM: (eliminating missing elements)
    print('\nSVM: (eliminating missing elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)

    # target = y_test.target
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    
    # Metrics
    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

    # SVM: (using all elements))
    print('\nSVM: (using all elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)

    # target = y_test.target
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    # Metrics
    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))
    
    pass

if __name__ == "__main__":
    main()
    