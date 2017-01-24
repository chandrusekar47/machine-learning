from collections import namedtuple
import sys
from math import *
from Data import *

DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'

def is_lambda(x):
    LAMBDA = lambda:0
    return isinstance(x, type(LAMBDA)) and LAMBDA.__name__ == x.__name__

def filter(data, feature, value_or_lambda):
    filtered = []
    filter_lambda = value_or_lambda
    if not is_lambda(value_or_lambda):
        filter_lambda = lambda x: x == value_or_lambda
    for record in data:
        if filter_lambda(record[feature]):
            filtered.append(record)
    return filtered

def probability(data, feature):
    num_total = float(len(data))
    if num_total == 0:
        return 0.0
    return len(filter(data, feature.feature, feature.value))/num_total

def entropy(data, feature):
    probab = probability(data, feature)
    if probab == 0 or probab == 1:
        return 0.0
    return -(probab * log(probab, 2) + (1 - probab) * log(1-probab, 2))

def InformationGain(data, f):
    positiveFeatureVal = FeatureVal(0, 'e')
    entropy_before_split = entropy(data, positiveFeatureVal)
    probab = probability(data, f)
    data_with_feature = filter(data, f.feature, f.value)
    data_without_feature = filter(data, f.feature, lambda x: x != f.value)
    info_gain = entropy_before_split - ((probab * entropy(data_with_feature, positiveFeatureVal)) + ((1-probab)*entropy(data_without_feature, positiveFeatureVal)))
    return info_gain

def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" % (prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')        
        
def ID3(data, features, MIN_GAIN=0.1):
    nPosNeg = [len(filter(data, 0, 'e')), len(filter(data, 0, 'p'))];
    if nPosNeg[0] == len(data):
        return DtNode(None, nPosNeg, 0, None, None)
    elif nPosNeg[0] == 0:
        return DtNode(None, nPosNeg, 0, None, None)
    max_info_gain = float("-inf")
    max_gain_feature = None

    for feature in features:
        info_gain = InformationGain(data, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_gain_feature = feature

    if max_info_gain < MIN_GAIN:
        return DtNode(None, nPosNeg, 0, None, None)
    left_child_data = filter(data, max_gain_feature.feature, max_gain_feature.value)
    right_child_data = filter(data, max_gain_feature.feature, lambda x:x != max_gain_feature.value)
    return DtNode(max_gain_feature, 
        nPosNeg, 
        max_info_gain, 
        ID3(left_child_data, features, MIN_GAIN),
        ID3(right_child_data, features, MIN_GAIN))

if __name__ == "__main__":
    train = MushroomData(sys.argv[1])
    dev = MushroomData(sys.argv[2])
    dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))
    
    PrintTree(dTree)

    print Accuracy(dTree, dev.data)
