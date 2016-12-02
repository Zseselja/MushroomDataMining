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
import copy
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

def plotROC(fprlist,tprlist,roc_aucList,clfstr , count , typelist):
    plt.figure()
    lw = count
    colorList = ['darkred', 'red' , 'salmon' , 'forestgreen' , 'lawngreen' , 'sage' , 'teal' , 'cyan', 'dodgerblue']
    for i in range(count):
        fpr = fprlist[i]
        tpr = tprlist[i]
        roc_auc = roc_aucList[i]


        # print "test"
        plt.plot(fpr[0], tpr[0], color=colorList[i], 
                 lw=lw, label=str(clfstr[0]) + " " + str(typelist[i]))
    # + '(area = %0.2f)' % roc_auc[i]
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.title("LinearSVC ROC",)
    plt.show()


        
# This program creates all the ROC curves for each classifier in our project. 
    

def main():
    # Get dataset from MushroomData
    n_classes = 1
    totalCount = 0
    data = MushroomData()
    typelist = ["missing data" , "all data" ,"stalk-root"]
    
    fprList = []
    tprList = []
    roc_aucList = []
    for j  in range(3):
        if j == 0:
            y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = True)
        elif j == 1:
            y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = False)
        elif j == 2:
            y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = False ,  ignore=['stalk-root'] )
            pass
        totalCount = 0

        X = np.array(data.X)
        y = np.array(data.y)
        y = label_binarize(y , classes= [1,-1])
        # print y
        # print X

        # y = label_binarize(y , classes= [1,-1])
       
        clf = LinearSVC()
        # clfList = [clf3]
        scoreList = [ "linearSVC"]
       
        
        


    # For each classifier 
        count  = 0
        for w  in range(1):
            # if w == 0:
            #     y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = True)
            # elif w == 1:
            #     y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = False)
            # elif w == 2:
            #     y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing = False ,  ignore=['stalk-root'] )
            #     pass
            # y = y_train
            # X = X_train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)
            y_score = clf.fit(X_train,y_train).decision_function(X_test)
            y_score = np.array([y_score])
            local = copy.deepcopy(y_test)
            # y_test = []
            # for x in local:
            #     y_test.append(X)
            # print y_test


            y_score = np.reshape(y_score , y_test.shape)
            # print y_score.shape
            # print y_test.shape

            y_test_prediction = clf.predict(X_test)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(0 , n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                # print fpr
                # print tpr
                roc_auc[i] = auc(fpr[i], tpr[i])
           

            fprList.append(fpr)
            tprList.append(tpr)
            roc_aucList.append(roc_auc)
            count +=1

         
            y_true = np.array(y_test)
           
            # print 'accuracy = %f' %( np.mean(( list(y_test)-y_test_prediction)==0))
            print(metrics.classification_report(y_true, y_test_prediction, target_names=data.class_labels, digits=6))
              

            
        totalCount += count
    plotROC(fprList,tprList,roc_aucList, scoreList , j+1 , typelist)

 

if __name__ == "__main__":
    main()
     
