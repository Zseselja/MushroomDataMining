
import urllib2

# Retreiving data from UCI mushroom dataset
dataUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
dataResponse = urllib2.urlopen(dataUrl)
attributes = dataResponse.read()
dataFile = open('MushDataSet.cvs' , 'w')
dataFile.write(attributes)


