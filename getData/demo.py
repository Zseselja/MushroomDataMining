#!/usr/bin/python

#Demo.py
#SENG474 Project
#Group: Alix Voorthuyzen, 
#       Alice Gibbons, 
#       Jason Curt, 
#       Matthew Clarkson, 
#       Zachary Seselja
#
#Purpose: Classify whether a mushroom is poisonous or not

import numpy as np 
import sys
import os
from sklearn.svm  import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from MushroomData import MushroomData


def main():
	data = MushroomData()

	y_test,X_test,y_train,X_train = data.get_datasets(eliminate_missing=True)

	clf = LinearSVC()
	clf.fit(X_train,y_train)

	classifiers = sys.argv
	del classifiers[0]

	test = (classifiers, )
	

	print test

	y_prediction = clf.predict(test)
	print y_prediction

if __name__ == "__main__":
    main()