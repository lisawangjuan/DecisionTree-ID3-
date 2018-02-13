from node import Node
import math
import copy
from parse import ProcessData
import random


def handleMissing(data):
    clean_data = list()
    attributes_unique_values = {}
    # get all unique values for each attribute
    for attribute in data[0].keys():
        values = [datum[attribute] for datum in data]
        unique_values = list(set(values))
        if '?' in unique_values:
            unique_values.remove('?')
            attributes_unique_values[attribute] = attributes_unique_values
    # Scan over the data and correct missing values
    for datum in data:
        for attribute, value in datum.iteritems():
            if value == '?':
                datum[attribute] = bestAttribute_value(data, attribute, datum['Classifier'], attributes_unique_values[attribute])
    return data


'''
find the most frequent attribute value, used for handling missing data
'''
def bestAttribute_value(data, attribute, classification, values):
    data = {}
    max_count = -1
    max_attribute_value = None
    for datum in data:
        if datum['Classifier'] == classification:
            curr_value = datum.get(attribute)
            if curr_value != '?':
                if curr_value in data.keys():
                    data[curr_value] += 1
                else:
                    data[curr_value] = 1
    for attribute_value, count in data.iteritems():
        if count > max_count:
            max_count = count
            max_attribute_value = attribute_value
    return max_attribute_value



'''
find the mojority value of Classifier attribute
'''
def majority(data):
    values = list(set([datum['Classifier'] for datum in data]))
    counts = [0] * len(values)
    for datum in data:
        for i in range(0, len(values)):
            if datum['Classifier'] == values[i]:
                counts[i] += 1
    index = counts.index(max(counts))
    return values[index]



''' calculate data entroy
'''
def entropy(classification_data):
    total = len(classification_data)
    pos = len([i for i, x in enumerate(classification_data) if x == classification_data[0]])
    probability = float(pos) / total
    entropy = 0.0
    try:
        return -probability * math.log(probability, 2) - (1 - probability) * math.log(1 - probability, 2)
    except ValueError as e:
        return entropy



''' calculate information gain of given attribute
'''
def information_gain(attribute, data, total_entropy):
    values = [datum[attribute] for datum in data]
    unique_values = list(set(values))
    classification_data_sets = [[] for i in range(0, len(unique_values))]
    for datum in data:
        attribute_value = datum[attribute]
        for i in range(0, len(unique_values)):
            if attribute_value == unique_values[i]:
                classification_data_sets[i].append(datum['Classifier'])
    weights = [float(len(classification_data_sets[i])) / float((len(data))) for i in
               range(0, len(classification_data_sets))]
    for data_set in classification_data_sets:
        if len(data_set) == 0:
            return 0  # we have noise
    entropies = [entropy(classification_data_sets[i]) for i in range(0, len(classification_data_sets))]
    return_entropy = total_entropy
    for i, val in enumerate(entropies):
        return_entropy -= weights[i] * val
    return return_entropy



''' return attribute with highest information gain
'''
def bestAttribute(data):
    attributes = data[0].keys()
    # removes key: Classifier as it is not an attribute
    attributes.remove('Classifier')
    best_attribute = None
    max_gain = 0
    classification_data = [datum['Classifier'] for datum in data]
    total_entropy = entropy(classification_data)

    for attribute in attributes:
        gain = information_gain(attribute, data, total_entropy)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    return best_attribute



'''
store branches information, later it can be used for pruning
'''
def get_branches(data, best_attribute):
    values = [datum[best_attribute] for datum in data]
    unique_values = list(set(values))
    branches = [[] for i in range(0, len(unique_values))]
    for datum in data:
        value = datum[best_attribute]
        for i in range(0, len(unique_values)):
            if value == unique_values[i]:
                branches[i].append(datum)
    return branches



''' check if the data fall in same classification
'''
def is_homogenous(data):
    base_Classifier = data[0]['Classifier']
    for datum in data:
        if datum['Classifier'] != base_Classifier:
            return False
    return True



def buidDecisionTree(data, default):
    data_clean = handleMissing(data)
    return id3(data_clean, default)


'''
build decision tree based on id3 algorithm
'''
def id3(data, default):
    # create root node
    root_node = Node()
    root_node.classifiers = list(set([datum['Classifier'] for datum in data]))
    # check if all classifications are the same
    if is_homogenous(data):
        root_node.label = data[0]['Classifier']
        return root_node
    # if no more attributes to split on, set the label or root_node to default
    elif len(data[0]) == 1:
        root_node.label = default
        return root_node
    else:
        # find best attribute to split
        best_attribute = bestAttribute(data)
        if best_attribute is None:
            root_node.label = default
            return root_node
        root_node.label = best_attribute
        root_node.branches = get_branches(data, best_attribute)
        for branch in root_node.branches:
            majority_branch = majority(branch)
            if len(branch) == 0:
                leaf_node = Node()
                leaf_node.label = majority_branch
                if len(root_node.children) == 0:
                    root_node.children[0] = leaf_node
                else:
                    keys = root_node.children.keys()
                    num_keys = len(root_node.children) - 1
                    index = keys[num_keys] + 1
                    root_node.children[index] = leaf_node
            else:
                if len(root_node.children) == 0:
                    root_node.children[0] = id3(branch, majority_branch)
                else:
                    keys = root_node.children.keys()
                    num_keys = len(root_node.children) - 1
                    index = keys[num_keys] + 1
                    root_node.children[index] = id3(branch, majority_branch)
        return root_node



def _prune(node, prune_label, data):
    node_list = [node]
    root_node = node
    while len(node_list) > 0:
        curr_node = node_list[0]
        if curr_node.label == prune_label:
            total_data = []
            for branch in curr_node.branches:
                for datum in branch:
                    total_data.append(datum['Classifier'])
            values = list(set(total_data))
            max_count = total_data.count(values[0])
            majority_classifier = values[0]
            for value in values:
                if total_data.count(value) > max_count:
                    max_count = total_data.count(value)
                    majority_classifier = value
            curr_node.label = majority_classifier
            curr_node.children = {}
            return root_node
        for key in curr_node.children.keys():
            if key in curr_node.children:
                node_list.append(curr_node.children[key])
        node_list.pop(0)
    return root_node



def find_max_prune_gain(pruning_scores_dict):
    keys = pruning_scores_dict.keys()
    max_gain = pruning_scores_dict[keys[0]]
    return_key = keys[0]
    for key in keys:
        if pruning_scores_dict[key] > max_gain:
            max_gain = pruning_scores_dict[key]
            return_key = key
    return return_key, max_gain



def get_prune_scores(node, data):
    scores = {}
    node_list = [node]
    root_node = node
    while len(node_list) > 0:
        prune_node = node_list[0]
        if prune_node.is_leaf():
            node_list.pop(0)
            continue
        scores[prune_node.label] = accuracy(_prune(copy.deepcopy(root_node), prune_node.label, data), data)
        for key in prune_node.children.keys():
            if key in prune_node.children:
                node_list.append(prune_node.children[key])
        node_list.pop(0)
    return scores


'''
Takes in a trained tree and a validation set of data.  Prunes nodes in order to improve accuracy on the validation data
'''
def prune(node, data):
    prune = True
    node_list = [node]
    visited = []
    root_node = node
    while prune and len(node_list) > 0:
        curr_node = node_list[0]
        base_acc = accuracy(root_node, data)
        if curr_node.is_leaf() or curr_node.label in visited:
            if curr_node in visited:
                for key in curr_node.children.keys():
                    if key in curr_node.children and not curr_node.children[key].label in visited:
                        node_list.append(curr_node.children[key])
            node_list.pop(0)
            continue
        visited.append(curr_node.label)
        scores = get_prune_scores(curr_node, data)
        max_score_label, max_score = find_max_prune_gain(scores)
        if max_score < base_acc:
            prune = False
            break
        root_node = _prune(node, max_score_label, data)
        node_list = [root_node]
    return root_node



'''
Takes in a trained tree and a test set of data.  Returns the accuracy (fraction
of data the tree classifies correctly).
'''
def accuracy(node, data):
    count = 0
    for datum in data:
        if datum['Classifier'] == predict(node, datum):
            count += 1
    return float(count) / float(len(data))



'''
Takes in a trained tree and a test set of data.  Returns the accuracy (fraction
of data the tree classifies correctly).
'''
def predict(node, datum):
    i = 0
    while node.children:
        attribute = node.label
        i += 1
        if datum[attribute] == node.branches[0][0][attribute]:
            node = node.children[0]
        else:
            node = node.children[1]
    return node.label



''' 
----------------------
------------------
TEST CASES 
------------------
----------------------
'''
def test_monk():
    from parse import ProcessData
    p = ProcessData()
    monk_train, header = p.readMonk("monk_train1.txt")
    monk_data, header2 = p.readMonk("monk_test1.txt")
    monk_valid, monk_test = p.splitData(monk_data)
    train = p.rebuildData(monk_train, header)
    valid = p.rebuildData(monk_valid, header)
    test = p.rebuildData(monk_test, header)

    tree = buidDecisionTree(train, "1")
    print "Building decision tree for monks' problem set...." + '\n'
    acc1 = accuracy(tree, test)
    acc2 = accuracy(tree, train)
    acc3 = accuracy(tree, valid)
    print "predict with test data before pruning: " + str(acc1)
    print "predict with train data before pruning: " + str(acc2)
    print "predict with valid data before pruning: " + str(acc3) + '\n'

    prune(tree, valid)
    print "pruning with valid data" + '\n'

    acc4 = accuracy(tree, test)
    acc5 = accuracy(tree, train)
    acc6 = accuracy(tree, valid)

    print "predict with test data after pruning: " + str(acc4)
    print "predict with train data after pruning: " + str(acc5)
    print "predict with valid data after pruning: " + str(acc6) + '\n\n\n'


def test_ecoli():
    from parse import ProcessData
    p = ProcessData()
    data, header = p.readEcoli("ecoli.txt")
    ecoli_data, ecoli_valid = p.splitData(data)
    ecoli_train, ecoli_test = p.splitData(ecoli_data)
    train = p.rebuildData(ecoli_train, header)
    valid = p.rebuildData(ecoli_valid, header)
    test = p.rebuildData(ecoli_test, header)

    tree = buidDecisionTree(train, 'T')
    print "Building decision tree for Ecoli set...." + '\n'

    acc1 = accuracy(tree, test)
    acc2 = accuracy(tree, train)
    acc3 = accuracy(tree, valid)
    print "predict with test data before pruning: " + str(acc1)
    print "predict with train data before pruning: " + str(acc2)
    print "predict with valid data before pruning: " + str(acc3) + '\n'

    prune(tree, valid)
    print "Pruning with valid data....." + '\n'

    acc4 = accuracy(tree, test)
    acc5 = accuracy(tree, train)
    acc6 = accuracy(tree, valid)
    print "predict with test data after pruning: " + str(acc4)
    print "predict with train data after pruning: " + str(acc5)
    print "predict with valid data after pruning: " + str(acc6) + '\n\n\n'


def test_mammpgraphic():
    from parse import ProcessData
    p = ProcessData()
    mammographic_data, header = p.readMammographic("mammographic-masses.txt")
    data, mammographic_test = p.splitData(mammographic_data)
    mammographic_train, mammographic_valid = p.splitData(data)

    train = p.rebuildData(mammographic_train, header)
    test = p.rebuildData(mammographic_test, header)
    valid = p.rebuildData(mammographic_valid, header)

    tree = buidDecisionTree(train, "0")
    print "Building decision tree for mammographic-masses set...." + '\n'

    acc1 = accuracy(tree, test)
    acc2 = accuracy(tree, train)
    acc3 = accuracy(tree, valid)
    print "predict with test data before pruning: " + str(acc1)
    print "predict with train data before pruning: " + str(acc2)
    print "predict with valid data before pruning: " + str(acc3) + '\n'

    prune(tree, valid)
    print "Pruning with valid data....." + '\n'

    acc4 = accuracy(tree, test)
    acc5 = accuracy(tree, train)
    acc6 = accuracy(tree, valid)
    print "predict with test data after pruning: " + str(acc4)
    print "predict with train data after pruning: " + str(acc5)
    print "predict with valid data after pruning: " + str(acc6) + '\n\n\n'


if __name__ == "__main__":
    test_monk()
    # test_ecoli()
    # test_mammpgraphic()
