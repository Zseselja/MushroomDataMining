#!/usr/bin/python

#decision_tree.py
#
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from MushroomData import MushroomData
from sklearn import metrics

if __name__ == "__main__":

    # Decision Tree: (with missing elements)
    print('Decision Tree: (eliminating missing elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    # Metrics
    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction) == 0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

    # Feature Importances 
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    y_pos = np.arange(len(importances))
    plt.bar(y_pos, importances[indices], align="center", alpha=0.5)
    plt.title("Feature Importance: Decision Tree (missing elements)")
    plt.ylabel('Importance')
    plt.xlabel('Attribute Number')
    plt.xticks(y_pos, indices)
    #plt.show()

    # Decision Tree: (using all elements)
    print('Decision Tree: (using all elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    # Metrics
    y_true = np.array(y_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))
    print(metrics.classification_report(y_true, y_prediction, target_names=data.class_labels, digits=6))

    # Feature Importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    y_pos = np.arange(len(importances))
    plt.bar(y_pos, importances[indices], align="center", alpha=0.5)
    plt.title("Feature Importance: Decision Tree (all elements)")
    plt.ylabel('Importance')
    plt.xlabel('Attribute Number')
    plt.xticks(y_pos, indices)
    #plt.show()

    # Weighted Decision Tree:
    #print('Elements weighted using priority of guidebook')

    # Four attributes were of extra importance when used for classification purposes.
    # These were gill-color(#9), veil-type(#16), ring-type(#19) and gill-attachment(#6)
    # These priorities were used along with the feature_importances_ of the classifiers to genererate a new decision tree

    #TODO: Create own decision tree using feature_importances and guidebook

