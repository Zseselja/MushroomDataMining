#!/usr/bin/python

# LinearSVC
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
from sklearn.svm  import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from MushroomData import MushroomData
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_fscore_support



def main():

    data = MushroomData()



    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    
    print('missing elements')

    clf = LinearSVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

# Forming plot 
    # plot_calibration_curve(clf , 'SVC', 1 ,  y_test, X_test,y_train,X_train)
    # plt.show()

    y_true = np.array(y_test)
    # print "macro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='macro'))+ "\n"
    # print "micro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='micro'))+ "\n"
    # print "weighted precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='weighted'))+ "\n"
    # print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))



    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    
    print('\nAll Elements')

    clf = LinearSVC()
    yscore = clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

# Forming plot 
    # plot_calibration_curve(clf , 'SVC', 1 ,  y_test, X_test,y_train,X_train)
    # plt.show()

    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))
    # print "macro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='macro'))+ "\n"
    # print "micro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='micro'))+ "\n"
    # print "weighted precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='weighted'))+ "\n"
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i in range(1):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yscore[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(np.ravel(y_test), np.ravel(yscore))
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])    
    plt.figure()
    plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

  


if __name__ == "__main__":
    main()
     