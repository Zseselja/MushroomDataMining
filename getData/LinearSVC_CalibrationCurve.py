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
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_fscore_support


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

    data = MushroomData()



    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)
    
    print('missing elements')

    clf = LinearSVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

# Forming plot 
    plot_calibration_curve(clf , 'SVC', 1 ,  y_test, X_test,y_train,X_train)
    plt.show()

    y_true = np.array(y_test)
    print "macro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='macro'))+ "\n"
    print "micro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='micro'))+ "\n"
    print "weighted precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='weighted'))+ "\n"
    print 'accuracy = %f' %( np.mean((list(y_test)-y_prediction)==0))



    data = MushroomData()
    y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=False)
    
    print('\nAll Elements')

    clf = LinearSVC()
    clf.fit(X_train,y_train)
    y_prediction = clf.predict(X_test)

# Forming plot 
    plot_calibration_curve(clf , 'SVC', 1 ,  y_test, X_test,y_train,X_train)
    plt.show()

    y_true = np.array(y_test)
    print "macro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='macro'))+ "\n"
    print "micro precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='micro'))+ "\n"
    print "weighted precision , recall , fscore = " + str(precision_recall_fscore_support(y_true, y_prediction, average='weighted'))+ "\n"

    
 
    print 'accuracy = %f' %( np.mean(( list(y_test)-y_prediction)==0))


if __name__ == "__main__":
    main()
     