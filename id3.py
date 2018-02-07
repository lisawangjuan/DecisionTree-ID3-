
from math import log
import operator
import numpy
from sklearn.cross_validation import train_test_split

# calculate entropy of data set
def entropy(data):
    entries = len(data)
    labels = {}
    for feat in data:
        label = feat[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels:
        probability = float(labels[key])/entries
        entropy -= probability * log(probability,2)
    return entropy

# split data set on a given feature
def split(data, axis, val):
    newData = []
    for feat in data:
        if feat[axis] == val:
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis+1:])
            newData.append(reducedFeat)
    return newData

# choose the best feature to split on
# def choose(data):
#     features = len(data[0]) - 1
#     baseEntropy = entropy(data)
#     bestInfoGain = 0.0;
#     bestFeat = -1
#     for i in range(features):
#         featList = [ex[i] for ex in data]
#         uniqueVals = set(featList)
#         newEntropy = 0.0
#         for value in uniqueVals:
#             newData = split(data, i, value)
#             probability = len(newData)/float(len(data))
#             newEntropy += probability * entropy(newData)
#         infoGain = baseEntropy - newEntropy
#         if (infoGain > bestInfoGain):
#             bestInfoGain = infoGain
#             bestFeat = i
#     return bestFeat

def choose(data, classifiers):
    baseEntropy = entropy(data)
    bestInfoGain = 0.0
    bestAttr = -1
    for i in xrange(len(data[0])):
      attrList = [attr[i] for attr in data]
      uniqueVals = set(attrList)
      newEntropy = 0.0
      for val in uniqueVals:
        newData = split(data, i, val)
        probability = len(newData) / float(len(data))
        newEntropy += probability * entropy(newData)
      infoGain = baseEntropy - newEntropy
      if (infoGain > bestInfoGain):
        bestInfoGain = infoGain
        bestFeat = i
      return bestFeat



# decides when to stop
def majority(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# main function
# def tree(data,labels):
def tree(data, classList, labels):
    # classList = [ex[-1] for ex in data]
    # print classList
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:
        return majority(classList)
    bestFeat = choose(data)
    print bestFeat
    bestFeatLabel = labels[bestFeat]
    print bestFeatLabel
    theTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    print labels
    featValues = [ex[bestFeat] for ex in data]
    print featValues
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        theTree[bestFeatLabel][value] = tree(split(data, bestFeat, value),subLabels)
    return theTree



def readData(filename):

    datafile = open(filename, 'r')
    data = []
    for line in datafile:
        row = []
        for attributes in line.rstrip("\n").split(","):
            row.append(attributes)
        data.append(row)
    return data

def splitIntoTrainAndTest(data):
    x = data.values[:, 0:-2]
    y = data.values[:, -1]
    # spliting the dataset into training and test
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    return x, y, x_train, y_train, x_test, y_test





print readData("ballonData")

# data = [["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"], ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
#         ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"], ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
#         ["Sunny", "Warm", "Normal", "Weak", "Warm", "Same", "No"]]
# labels = ["Sky", "Air-Temp", "Humidity", "Wind", "Water", "Forecast"]


# data = [["T", "T", "+"], ["T", "T", "+"], ["T", "F", "-"], ["F", "F", "+"], ["F", "T", "-"], ["F", "T", "-"]]
# labels = ["a1", "a2"]


data = [["YELLOW", "SMALL", "STRETCH", "ADULT", "T"],
        # ["YELLOW","SMALL","STRETCH","CHILD","T"],
        # ["YELLOW","SMALL","DIP","ADULT","T"],
        # ["YELLOW","SMALL","DIP","CHILD","T"],
        ["YELLOW","SMALL","STRETCH","ADULT","T"],
        # ["YELLOW","SMALL","STRETCH","CHILD","T"],
        # ["YELLOW","SMALL","DIP","ADULT","T"],
        ["YELLOW","SMALL","DIP","CHILD","T"],
        # ["YELLOW","LARGE","STRETCH","ADULT","F"],
        # ["YELLOW","LARGE","STRETCH","CHILD","F"],
        ["YELLOW","LARGE","DIP","ADULT","F"],
        # ["YELLOW","LARGE","DIP","CHILD","F"],
        # ["PURPLE","SMALL","STRETCH","ADULT","F"],
        # ["PURPLE","SMALL","STRETCH","CHILD","F"],
        ["PURPLE","SMALL","DIP","ADULT","F"],
        # ["PURPLE","SMALL","DIP","CHILD","F"],
        # ["PURPLE","LARGE","STRETCH","ADULT","F"],
        ["PURPLE","LARGE","STRETCH","CHILD","F"],
        # ["PURPLE","LARGE","DIP","ADULT","F"],
        ["PURPLE","LARGE","DIP","CHILD","F"]]
labels = ["Color", "Size", "Shape", "Age"]


# print tree(data, labels)