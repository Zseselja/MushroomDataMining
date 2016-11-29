#!/usr/bin/python

# LinearSVC.py
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
from sklearn.svm  import LinearSVC
from MushroomData import MushroomData
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics



def main():

    # Linear SVC: (eliminating missing elements)
    print('\nLinear SVC: (eliminating missing elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    clf = LinearSVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    # Metrics
    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

    # Linear SVC: (using all elements)
    print('\nLinear SVC: (using all elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    clf = LinearSVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    # Metrics
    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))


if __name__ == "__main__":
    main()
     