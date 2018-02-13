import numpy as np
from sklearn.model_selection import train_test_split


# process data in the way we need
class ProcessData(object):
    # read data from file and convert it to array
    def readMonk(self, filename):

        datafile = open(filename, 'r')
        header = datafile.next().rstrip("\n").split(" ")
        data = []
        for line in datafile:
            row = line.rstrip("\n").split(" ")
            data.append(row[:-1])
        return data, header

    def readEcoli(self, filename):
        datafile = open(filename, 'r')
        header = datafile.next().rstrip("\n").split(" ")
        data = []
        for line in datafile:
            row = line.rstrip("\n").split(" ")
            data.append(row)
        return data, header

    def readMammographic(self, filename):
        datafile = open(filename, 'r')
        header = datafile.next().rstrip("\n").split(",")
        print header
        data = []
        for line in datafile:
            row = line.rstrip("\n").split(",")
            data.append(row)
        return data, header

    def splitData(self, data):
        # split into training and testing datasets
        data = np.array(data)
        train, test = train_test_split(data, test_size=0.3, random_state=330)
        return train.tolist(), test.tolist()

    # combine attributes parts and classifiers part together, so later can be used to build decision tree
    def rebuildData(self, data, header):
        output = []
        for row in data:
            output.append(dict(zip(header, row)))
        return output
