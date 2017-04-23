import numpy as np
import re

def logExpSum(exponent_terms, product_terms):
    max_values = np.max(exponent_terms, axis = 1)
    max_deducted_values = exponent_terms - max_values.reshape(exponent_terms.shape[0], 1)
    log_term = (np.exp(max_deducted_values) * product_terms).sum(axis = 1)
    return max_values + np.log(log_term)

def readTrue(filename='wine-true.data'):
    f = open(filename)
    labels = []
    splitRe = re.compile(r"\s")
    for line in f:
        labels.append(int(splitRe.split(line)[0]))
    return labels

#########################################################################
#Reads and manages data in appropriate format
#########################################################################
class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows,self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    #Computers the range of each column (returns a list of min-max tuples)
    def Range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1,self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min,max))
        return ranges

    def as_numpy_array(self):
        return np.array(self.data)

    def __getitem__(self,row):
        return self.data[row]
