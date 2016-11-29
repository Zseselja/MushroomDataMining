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

def plotROC(fpr,tpr,roc_auc,clfstr , count):
    # quit()
    plt.figure()
    lw = count
    typelist = ["missing data" , "missing data" ,"missing data" ,"all data" ,"all data" ,"all data" , "stalk-root" ,"stalk-root","stalk-root"]
    colorList = ['darkred', 'red' , 'salmon' , 'forestgreen' , 'lawngreen' , 'sage' , 'teal' , 'cyan', 'dodgerblue']
    for i in range(count):
        plt.plot(fpr[i], tpr[i], color=colorList[i],
                 lw=lw, label=str(clfstr[i]) + " " + str(typelist[i])+ '(area = %0.2f)' % roc_auc[i])
   
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.title("All Classifiers")
    plt.show()


        
# This program creates all the ROC curves for each classifier in our project. 
    

def main():
    # Get dataset from MushroomData
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = True)
    totalCount = 0

    X = np.array(data.X)
    y = np.array(data.y)
    y = label_binarize(y , classes= [1,-1])
  
    clf1 = GaussianNB()
    clf2 = svm.SVC()
    clf3 = LinearSVC()
    clfList = [clf1,clf2,clf3]
    scoreList = [ "GaussianNB" , "SVM" , "linearSVC"]
    
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
   
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)

    fprList = []
    tprList = []
    roc_aucList = []
    # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
    #                              random_state=random_state))
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    # print n_classes
# For each classifier 
    count  = 0
    for clf  in clfList:
       
        
        clf.fit(X_train,y_train)
        y_test_prediction = clf.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
       
        fpr, tpr, _ = roc_curve(y_test, y_test_prediction)
        roc_auc = auc(fpr, tpr)

        fprList.append(fpr)
        tprList.append(tpr)
        roc_aucList.append(roc_auc)
        

        # plotROC(fpr,tpr,roc_auc,clfname.name)
       
        y_true = np.array(y_test)
       
        print scoreList[count]
        print 'accuracy = %f' %( np.mean(( list(y_test)-y_test_prediction)==0))
        print(metrics.classification_report(y_true, y_test_prediction, target_names=data.class_labels, digits=6))
          

        count +=1
    totalCount += count
# ---------------------------------------------------------------------------------------------
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = False)
    X = np.array(data.X)
    y = np.array(data.y)
    y = label_binarize(y , classes= [1,-1])
  
    clf1 = GaussianNB()
    clf2 = svm.SVC()
    clf3 = LinearSVC()
    clfList = [clf1,clf2,clf3]
    scoreList.extend([ "GaussianNB" , "SVM" , "linearSVC"])
    
    
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
   
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    count  = 0
    for clf  in clfList:
        clf.fit(X_train,y_train)
        y_test_prediction = clf.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
       
        fpr, tpr, _ = roc_curve(y_test, y_test_prediction)
        roc_auc = auc(fpr, tpr)

        fprList.append(fpr)
        tprList.append(tpr)
        roc_aucList.append(roc_auc)
        

        # plotROC(fpr,tpr,roc_auc,clfname.name)
       
        y_true = np.array(y_test)
       
        print scoreList[count]
        print 'accuracy = %f' %( np.mean(( list(y_test)-y_test_prediction)==0))
        print(metrics.classification_report(y_true, y_test_prediction, target_names=data.class_labels, digits=6))
          
        count +=1
    totalCount += count
    
# ------------------------------------------------------------------------------------------------
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False, ignore=['stalk-root'])
    

    X = np.array(data.X)
    y = np.array(data.y)
    y = label_binarize(y , classes= [1,-1])
  
    clf1 = GaussianNB()
    clf2 = svm.SVC()
    clf3 = LinearSVC()
    clfList = [clf1,clf2,clf3]
    scoreList.extend([ "GaussianNB" , "SVM" , "linearSVC"])
    
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
   
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    count  = 0
    for clf  in clfList:
        clf.fit(X_train,y_train)
        y_test_prediction = clf.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
       
        fpr, tpr, _ = roc_curve(y_test, y_test_prediction)
        roc_auc = auc(fpr, tpr)
        
        fprList.append(fpr)
        tprList.append(tpr)
        roc_aucList.append(roc_auc)




        # plotROC(fpr,tpr,roc_auc,clfname.name)
       
        y_true = np.array(y_test)
       
        print scoreList[count]
        print 'accuracy = %f' %( np.mean(( list(y_test)-y_test_prediction)==0))
        print(metrics.classification_report(y_true, y_test_prediction, target_names=data.class_labels, digits=6))
        count +=1   
    totalCount += count
    plotROC(fprList,tprList,roc_aucList, scoreList , totalCount)

    
 


if __name__ == "__main__":
    main()
     
