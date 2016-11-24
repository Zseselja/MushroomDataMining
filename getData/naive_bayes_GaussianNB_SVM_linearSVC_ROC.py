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
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, Series

from numpy import shape
from sklearn import svm , metrics
from sklearn.svm  import LinearSVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from MushroomData import MushroomData
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support , roc_curve , auc

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

def plotROC(fpr,tpr,roc_auc,clfstr):
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.title(clfstr)
    plt.show()

# Plot calibration curve for Gaussian Naive Bayes
# plot_calibration_curve(GaussianNB(), "Naive Bayes", 1 , y_test,X_test,y_train,X_train)
class score(object):
    """docstring for score"""
    def __init__(self , name):
        self.name = name
    def addData(self , precision , recall , fscore):
        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        
# This program creates all the ROC curves for each classifier in our project. 
def main():
    # Get dataset from MushroomData
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = True)
  
    X = np.array(data.X)
    y = np.array(data.y)
   
    y = label_binarize(y , classes= [1,-1])
  
   
    # n_classes = y.shape[1]

    clf1 = GaussianNB()
    clf2 = svm.SVC()
    clf3 = LinearSVC()
    clfList = [clf1,clf2,clf3]
    scoreList = [ score("GaussianNB") , score("SVM") , score("linearSVC")]
    

    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    # print random_state
    # print n_samples
    # print n_features
   
# Need to figure out how to add noise
   
    # X_train = np.c_[X_train, random_state.randn(len(X_train), 200 * n_features)]
    # print len(y)
    # print len(X)
    # # print len(X)
    # # print len(y)
    # seperates them in half
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.5,random_state=0)

    # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
    #                              random_state=random_state))
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    

    
    # print n_classes
# For each classifier 
    count  = 0
    for clf  in clfList:
        clfname = scoreList[count]
        count +=1
        clf.fit(X_train,y_train)
         # y_score = y_score.decision_function(X_test)

        y_test_prediction = clf.predict(X_test)
        # for val in np.array(y_test_prediction):
        #     if val == -1:
        #         val = 0
        #     newlist = [val]
        #     y_vals.append(newlist)

        # print y_vals    
        # figure out ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # fpr, tpr, thresholds = metrics.roc_curve(y_prediction, y_true, pos_label=2)
        # for i in range(n_classes ):
            # print i            
        # print y_score[:, i]
        # print y_test[:, i]
            # print np.array(y_test_prediction)
            
        fpr, tpr, _ = roc_curve(y_test, y_test_prediction)
        roc_auc = auc(fpr, tpr)
            # print fpr
            # print tpr

        plotROC(fpr,tpr,roc_auc,clfname.name)
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prediction.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        print 'accuracy = %f' %( np.mean(( list(y_test)-y_test_prediction)==0))
   
 


if __name__ == "__main__":
    main()
     
