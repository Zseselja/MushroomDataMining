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
from sklearn import tree
from MushroomData import MushroomData
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("testing data class")

    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    
    print('missing elements')

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    y_pos = np.arange(len(importances))
    plt.bar(y_pos, importances[indices], align="center", alpha=0.5)
    plt.title("Feature Importance: Decision Tree (missing elements)")
    plt.ylabel('Importance')
    plt.xlabel('Attribute Number')
    plt.xticks(y_pos, indices)
    plt.show()


    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    
    print('All Elements')

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    y_pos = np.arange(len(importances))
    plt.bar(y_pos, importances[indices], align="center", alpha=0.5)
    plt.title("Feature Importance: Decision Tree (all elements)")
    plt.ylabel('Importance')
    plt.xlabel('Attribute Number')
    plt.xticks(y_pos, indices)
    plt.show()

    # Element Weights:
    #print('Elements weighted using priority of guidebook')

    # Four attributes were of extra importance when used for classification purposes.
    # These were gill-color(#9), veil-type(#16), ring-type(#19) and gill-attachment(#6)
    # These priorities were used along with the feature_importances_ of the classifiers to genererate a new decision tree

    #TODO: Create own decision tree using feature_importances and guidebook

