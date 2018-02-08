
from math import log
import operator
import numpy as np
from sklearn.model_selection import train_test_split


# build a decision tree using ID3 algorithm
class DecisionTree(object):

    # calculate entropy of data set
    def entropy(self, data):
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
    def split(self, data, axis, val):
        newData = []
        for feat in data:
            if feat[axis] == val:
                reducedFeat = feat[:axis]
                reducedFeat.extend(feat[axis+1:])
                newData.append(reducedFeat)
        return newData

    # choose the best feature to split on
    def choose(self, data):
        features = len(data[0]) - 1
        baseEntropy = self.entropy(data)
        bestInfoGain = 0.0;
        bestFeat = -1
        for i in range(features):
            featList = [ex[i] for ex in data]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                newData = self.split(data, i, value)
                probability = len(newData)/float(len(data))
                newEntropy += probability * self.entropy(newData)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeat = i
        return bestFeat

    # def choose(data, classifiers):
    #     baseEntropy = entropy(data)
    #     bestInfoGain = 0.0
    #     bestAttr = -1
    #     for i in xrange(len(data[0])):
    #       attrList = [attr[i] for attr in data]
    #       uniqueVals = set(attrList)
    #       newEntropy = 0.0
    #       for val in uniqueVals:
    #         newData = split(data, i, val)
    #         probability = len(newData) / float(len(data))
    #         newEntropy += probability * entropy(newData)
    #       infoGain = baseEntropy - newEntropy
    #       if (infoGain > bestInfoGain):
    #         bestInfoGain = infoGain
    #         bestFeat = i
    #       return bestFeat



    # decides when to stop
    def majority(self, classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    # main function
    def buildTree(self, data,labels):
        classList = [ex[-1] for ex in data]
        # print classList
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(data[0]) == 1:
            return self.majority(classList)
        bestFeat = self.choose(data)
        # print bestFeat
        bestFeatLabel = labels[bestFeat]
        # print bestFeatLabel
        theTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        # print labels
        featValues = [ex[bestFeat] for ex in data]
        # print featValues
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            theTree[bestFeatLabel][value] = self.buildTree(self.split(data, bestFeat, value),subLabels)
        return theTree


# process data in the way we need
class ProcessData(object):
    # read data from file and convert it to array
    def readData(self,filename):

        datafile = open(filename, 'r')
        data = []
        for line in datafile:
            row = []
            for attributes in line.rstrip("\n").split(","):
                row.append(attributes)
            data.append(row)
        return data

    # split data into training and test sets
    def splitIntoTrainAndTest(self, data):
        data = np.array(data)
        x = data[:, 0:-2]
        y = data[:, -1]
        # spliting the dataset into training and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)
        return x_train, y_train, x_test, y_test

    # combine attributes parts and classifiers part together, so later can be used to build decision tree
    def rebuildData(self, data1, data2):
        # a = data1.tolist()
        # b = data2.tolist()
        # for i in xrange(len(a)):
        #     a[i].extend(b[i])
        # return a
        return np.column_stack((data1, data2)).tolist()


# # use decision tree built on training data to predict on test data and calculate the accuracy
# class Verification(object):
#
#     def prediction(self, tree, test):
#         ans = []
#








if __name__ == "__main__":
    p = ProcessData()
    data = p.readData("ballon.txt")
    x_train, y_train, x_test, y_test = p.splitIntoTrainAndTest(data)
    data = p.rebuildData(x_train, y_train)
    # testData = rebuildData(x_test, y_test)
    # print testData

    weatherLabels = ["Sky", "Air-Temp", "Humidity", "Wind", "Water", "Forecast"]
    ballonLabels = ["Color", "Size", "Shape", "Age"]

    d = DecisionTree()
    print d.buildTree(data, ballonLabels)
    print
    print x_test

    print
    print y_test