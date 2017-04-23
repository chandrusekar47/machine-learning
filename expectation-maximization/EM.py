#!/usr/bin/python

#########################################################
# CSE 5523 starter code (HW#5)
# Alan Ritter
#########################################################

import random
import math
import sys
import re
import numpy as np
import scipy.stats
from util import *
import matplotlib.pyplot as plt
from collections import Counter

#GLOBALS/Constants
VAR_INIT = 1
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#########################################################################
#Computes EM on a given data set, using the specified number of clusters
#self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################
class EM:
    def __init__(self, data, nClusters):
        #Initialize parameters randomly...
        random.seed()
        self.parameters = []
        self.priors = []        #Cluster priors
        self.nClusters = nClusters
        self.data = data.as_numpy_array()
        self.nDims = self.data.shape[1]
        self.nRecords = self.data.shape[0]
        ranges = data.Range()
        for i in range(nClusters):
            p = []
            for j in range(data.nCols):
                #Randomly initalize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), ranges[j][1] - ranges[j][0]))
            self.parameters.append(p)
        #Initialize priors uniformly
        for c in range(nClusters):
            self.priors.append(1/float(nClusters))
        self.priors = np.array(self.priors)

    def assign_classes(self, predicted_cluster_numbers, true_cluster_numbers):
        unique_clusters = np.unique(predicted_cluster_numbers)
        classes = np.zeros(len(predicted_cluster_numbers))
        cluster_to_class = {}
        for cluster in unique_clusters:
            points_in_cluster = predicted_cluster_numbers == cluster
            majority_class = scipy.stats.mode(true_cluster_numbers[points_in_cluster]).mode[0]
            classes[points_in_cluster] = majority_class
            cluster_to_class[cluster] = majority_class
        return (classes, cluster_to_class)

    def LogLikelihood(self, data):
        overallLikelihood = 0.0
        const_term = pow(np.sqrt(2*np.pi), self.nDims)
        for i, record in enumerate(data):
            exponent_terms = np.zeros(self.nClusters)
            product_terms = np.zeros(self.nClusters)
            for k in xrange(0,self.nClusters):
                product_of_vars = 1.0
                exp_term = 0.0
                for j in range(0, self.nDims):
                    mean_j = self.parameters[k][j][0]
                    var_j = self.parameters[k][j][1]
                    product_of_vars *= np.sqrt(var_j)
                    exp_term -= (record[j] - mean_j)**2/(2 * (var_j))
                exponent_terms[k] = exp_term
                product_terms[k] = self.priors[k] * 1.0 /(const_term * product_of_vars)
            x_max = np.max(exponent_terms)
            exponent_terms = exponent_terms - x_max
            overallLikelihood += x_max + np.log(np.sum(np.exp(exponent_terms) * product_terms))
        return overallLikelihood

    def compute_posterior(self, data):
        nRecords = len(data)
        y_posterior = np.zeros((nRecords, self.nClusters))
        for k in range(0, self.nClusters):
            for j in range(0, nRecords):
                for i in range(0, self.nDims):
                    mean_i = self.parameters[k][i][0]
                    var_i = self.parameters[k][i][1]
                    den_const_term = 2 * (var_i)
                    x_ji = data[j][i]
                    y_posterior[j][k] += (-0.5 * np.log(var_i)) - ((x_ji - mean_i)**2/den_const_term)
                y_posterior[j][k] += np.log(self.priors[k])
            y_posterior[:, k] = np.exp(y_posterior[:, k])
        return y_posterior / y_posterior.sum(axis = 1).reshape(nRecords, 1)

    #Compute marginal distributions of hidden variables
    def Estep(self):
        self.y_posterior = self.compute_posterior(self.data)

    #Update the parameter estimates
    def Mstep(self):
        y_posterior_sums = self.y_posterior.sum(axis = 0)
        if np.any(y_posterior_sums == 0):
            print("Empty clusters generated. Re-run the program to choose different starting points.")
            exit(-1)
        for k in range(0, self.nClusters):
            mean_k = np.sum(self.data * self.y_posterior[:, k].reshape(self.nRecords, 1), axis = 0)/y_posterior_sums[k]
            var_k = np.array(map(lambda x: max(0.00001, x), (np.sum((self.data**2) * self.y_posterior[:, k].reshape(self.nRecords, 1), axis = 0)/y_posterior_sums[k]) - mean_k**2))
            self.priors[k] = y_posterior_sums[k]/self.nRecords
            for j in xrange(0, self.nDims):
                self.parameters[k][j] = (mean_k[j], var_k[j])

    def posterior_to_cluster_number(self, cluster_posteriors):
        return np.argmax(cluster_posteriors, axis = 1)

    def cluster_number_to_class(self, cluster_numbers, cluster_class_mapping):
        return np.array(map(lambda x: cluster_class_mapping[x], cluster_numbers)).astype('float')

    def calc_confusion_matrix(self, actual_class, predicted_class):
        unique_classes = np.unique(actual_class).tolist()
        unique_classes.sort()
        confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)))
        for ind, klazz in enumerate(unique_classes):
            counts = Counter(predicted_class[actual_class == klazz])
            for klazz, count in counts.iteritems():
                confusion_matrix[ind][unique_classes.index(klazz)] += count
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis = 1).reshape(len(unique_classes), 1)
        return confusion_matrix

    def Run(self, maxsteps=100, testData=None, trueLabels = []):
        #TODO: Implement EM algorithm
        trainLikelihood = -np.inf
        testLikelihood = 0.0
        num_steps = 0
        trainLikelihoodValues = []
        testLikelihoodValues = []
        while num_steps < maxsteps:
            newLikelihood = self.LogLikelihood(self.data)
            testLikelihood = 0 if testData == None else self.LogLikelihood(testData)
            testLikelihoodValues.append(testLikelihood)
            trainLikelihoodValues.append(newLikelihood)
            if num_steps > 0:
                percent_change = abs(newLikelihood - trainLikelihood) * 100/trainLikelihood
                if abs(percent_change) < 0.001:
                    trainLikelihood = newLikelihood
                    break
            trainLikelihood = newLikelihood
            self.Estep()
            self.Mstep()
            num_steps += 1
        (predicted_labels, cluster_class_mapping) = self.assign_classes(self.posterior_to_cluster_number(self.y_posterior), trueLabels)
        confusion_matrix = self.calc_confusion_matrix(trueLabels, predicted_labels)
        test_predicted_class = self.cluster_number_to_class(self.posterior_to_cluster_number(self.y_posterior), cluster_class_mapping)
        overall_accuracy = float(np.argwhere(predicted_labels == np.array(trueLabels)).shape[0]) / float(len(trueLabels))
        return (trainLikelihood, testLikelihood, trainLikelihoodValues, testLikelihoodValues, confusion_matrix, test_predicted_class, overall_accuracy)

def run_em_for_different_k():
    d = Data('data/wine.train')
    testData = Data('data/wine.test')
    trueLabels = np.array(readTrue("data/wine-true.data"))
    k_values = xrange(1, 11)
    values = []
    for k in k_values:
        em = EM(d, k)
        (trainLikelihood, testLikelihood, _, _, _, _, overall_accuracy) = em.Run(100, testData = testData, trueLabels = trueLabels)
        values.append([trainLikelihood, testLikelihood, overall_accuracy])
    plt.plot(k_values, [x[0] for x in values])
    plt.title("Training data likelihood vs number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Log likelihood")
    plt.show()
    plt.plot(k_values, [x[1] for x in values])
    plt.title("Test data likelihood vs number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Log likelihood")
    plt.show()
    plt.plot(k_values, [x[2] for x in values])
    plt.title("Training data accuracy vs number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Accuracy")
    plt.show()


def main():
    # run_em_for_different_k()
    # exit()
    d = Data('data/wine.train')
    testData = Data('data/wine.test')
    trueLabels = np.array(readTrue("data/wine-true.data"))

    if len(sys.argv) > 1:
        e = EM(d, int(sys.argv[1]))
    else:
        e = EM(d, 3)
    (trainLikelihood, testLikelihood, trainLikelihoods, testLikelihoods, confusion_matrix, test_predicted_class, overall_accuracy) = e.Run(100, testData = testData, trueLabels = trueLabels)
    print("Accuracy: " + str(overall_accuracy))
    print("Confusiong matrix: ")
    print(confusion_matrix)
    print(str(trainLikelihood) + ", " + str(testLikelihood))
    print("True cluster number prediction for test data")
    print(test_predicted_class)
    # plt.plot(xrange(1, len(trainLikelihoods[1:]) + 1), trainLikelihoods[1:])
    # plt.title("Training data likelihood vs iteration")
    # plt.xlabel("Iteration number")
    # plt.ylabel("Log likelihood")
    # plt.show()
    # plt.plot(xrange(1, len(testLikelihoods[1:]) + 1), testLikelihoods[1:])
    # plt.title("Test data likelihood vs iteration")
    # plt.xlabel("Iteration number")
    # plt.ylabel("Log likelihood")
    # plt.show()

if __name__ == "__main__":
    main()