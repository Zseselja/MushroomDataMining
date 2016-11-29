#!/usr/bin/python

# decision_tree.py
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def plot_calibration_curve(est, name, fig_index, y_test,X_test,y_train,X_train):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_pred.max())
        # print("%s:" % name)
        # print("\tBrier: %1.3f" % (clf_score))
        # print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        # print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        # print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


def main():

    # Decision Tree: (eliminating missing elements)
    print('\nDecision Tree: (eliminating missing elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)
    
    # Calibration Curve Plot 
    plot_calibration_curve(clf , 'Decision Tree', 1 , y_test, X_test, y_train, X_train)
    plt.show()

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
    print('\nDecision Tree: (using all elements)')
    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

    # Calibration Curve Plot 
    plot_calibration_curve(clf , 'Decision Tree', 1 ,  y_test, X_test,y_train,X_train)
    plt.show()

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

    #TODO: [AG] Create own decision tree using feature_importances and guidebook

if __name__ == "__main__":
    main()
