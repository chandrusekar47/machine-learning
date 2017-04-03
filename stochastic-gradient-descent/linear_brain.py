import scipy.io as sio
from scipy.misc import logsumexp
import numpy as np
from decimal import Decimal

starplus = sio.loadmat("data-starplus-04847-v7.mat")

###########################################
metadata = starplus['meta'][0,0]
#meta.study gives the name of the fMRI study
#meta.subject gives the identifier for the human subject
#meta.ntrials gives the number of trials in this dataset
#meta.nsnapshots gives the total number of images in the dataset
#meta.nvoxels gives the number of voxels (3D pixels) in each image
#meta.dimx gives the maximum x coordinate in the brain image. The minimum x coordinate is x=1. meta.dimy and meta.dimz give the same information for the y and z coordinates.
#meta.colToCoord(v,:) gives the geometric coordinate (x,y,z) of the voxel corresponding to column v in the data
#meta.coordToCol(x,y,z) gives the column index (within the data) of the voxel whose coordinate is (x,y,z)
#meta.rois is a struct array defining a few dozen anatomically defined Regions Of Interest (ROIs) in the brain. Each element of the struct array defines on of the ROIs, and has three fields: "name" which gives the ROI name (e.g., 'LIFG'), "coords" which gives the xyz coordinates of each voxel in that ROI, and "columns" which gives the column index of each voxel in that ROI.
#meta.colToROI{v} gives the ROI of the voxel corresponding to column v in the data.
study      = metadata['study']
subject    = metadata['subject']
ntrials    = metadata['ntrials'][0][0]
nsnapshots = metadata['nsnapshots'][0][0]
dimx       = metadata['dimx'][0][0]
colToCoord = metadata['colToCoord']
coordToCol = metadata['coordToCol']
rois       = metadata['rois']
colToROI   = metadata['colToROI']
###########################################

###########################################
info = starplus['info'][0]
#info: This variable defines the experiment in terms of a sequence of 'trials'. 'info' is a 1x54 struct array, describing the 54 time intervals, or trials. Most of these time intervals correspond to trials during which the subject views a single picture and a single sentence, and presses a button to indicate whether the sentence correctly describes the picture. Other time intervals correspond to rest periods. The relevant fields of info are illustrated in the following example:
#info(18) mint: 894 maxt: 948 cond: 2 firstStimulus: 'P' sentence: ''It is true that the star is below the plus.'' sentenceRel: 'below' sentenceSym1: 'star' sentenceSym2: 'plus' img: sap actionAnswer: 0 actionRT: 3613
#info.mint gives the time of the first image in the interval (the minimum time)
#info.maxt gives the time of the last image in the interval (the maximum time)
#info.cond has possible values 0,1,2,3. Cond=0 indicates the data in this segment should be ignored. Cond=1 indicates the segment is a rest, or fixation interval. Cond=2 indicates the interval is a sentence/picture trial in which the sentence is not negated. Cond=3 indicates the interval is a sentence/picture trial in which the sentence is negated.
#info.firstStimulus: is either 'P' or 'S' indicating whether this trail was obtained during the session is which Pictures were presented before sentences, or during the session in which Sentences were presented before pictures. The first 27 trials have firstStimulus='P', the remained have firstStimulus='S'. Note this value is present even for trials that are rest trials. You can pick out the trials for which sentences and pictures were presented by selecting just the trials trials with info.cond=2 or info.cond=3.
#info.sentence gives the sentence presented during this trial. If none, the value is '' (the empty string). The fields info.sentenceSym1, info.sentenceSym2, and info.sentenceRel describe the two symbols mentioned in the sentence, and the relation between them.
#info.img describes the image presented during this trial. For example, 'sap' means the image contained a 'star above plus'. Each image has two tokens, where one is above the other. The possible tokens are star (s), plus (p), and dollar (d).
#info.actionAnswer: has values -1 or 0. A value of 0 indicates the subject is expected to press the answer button during this trial (either the 'yes' or 'no' button to indicate whether the sentence correctly describes the picture). A value of -1 indicates it is inappropriate for the subject to press the answer button during this trial (i.e., it is a rest, or fixation trial).
#info.actionRT: gives the reaction time of the subject, measured as the time at which they pressed the answer button, minus the time at which the second stimulus was presented. Time is in milliseconds. If the subject did not press the button at all, the value is 0.
###########################################

###########################################
data = starplus['data']
#data: This variable contains the raw observed data. The fMRI data is a sequence of images collected over time, one image each 500 msec. The data structure 'data' is a [54x1] cell array, with one cell per 'trial' in the experiment. Each element in this cell array is an NxV array of observed fMRI activations. The element data{x}(t,v) gives the fMRI observation at voxel v, at time t within trial x. Here t is the within-trial time, ranging from 1 to info(x).len. The full image at time t within trial x is given by data{x}(t,:).
#Note the absolute time for the first image within trial x is given by info(x).mint.
###########################################


def LogisticLoss(X, Y, W, lmda):
    value = np.exp(-1 * Y * X.dot(W));
    return (np.log2(1 + value)).sum() + lmda * (W**2).sum()

def LogisticGradient(X, Y, W, lmda):
    expo = np.exp((-Y * X.dot(W)).astype('float64'))
    return ((expo/(1+expo)) * -Y).dot(X) + 2 * lmda * W

def HingeLoss(X, Y, W, lmda):
    return np.maximum(0, 1 - (Y * X.dot(W))).sum() + lmda * (W**2).sum()

def HingeGradient(X, Y, W, lmda):
    func_val = 1 - (Y * X.dot(W))
    return (-Y[func_val > 0].dot(X[func_val > 0, :])) + 2 * lmda * W

def StochasticGradientDescent(X, Y, W, maxIter, learningRate, lmda, gradientFunc, lossFunc):
    loss_diff = 1
    iters = 0
    no_records = len(X)
    while abs(loss_diff) > 0.0001 and iters < maxIter:
        loss_before = lossFunc(X, Y, W, lmda)
        for record_num in xrange(1,no_records):
            record = X[record_num]
            actual_output = Y[record_num]
            gradient = gradientFunc(np.array([record]), np.array([actual_output]), W, lmda)
            W = W - learningRate*gradient
        loss_after = lossFunc(X, Y, W, lmda)
        # print("Loss at the end of iteration %s: %s"%(iters, loss_after))
        loss_diff = loss_before - loss_after
        iters += 1
    return W

def SgdHinge(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1]).astype('float64')
    return StochasticGradientDescent(X, Y, W, maxIter, learningRate, lmda, HingeGradient, HingeLoss)

def SgdLogistic(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1]).astype('float64')
    return StochasticGradientDescent(X, Y, W, maxIter, learningRate, lmda, LogisticGradient, LogisticLoss)

def crossValidation(X, Y, SGD, lmda, learningRate, maxIter=100, sample=range(20)):
    #Leave one out cross validation accuracy
    nCorrect   = 0.
    nIncorrect = 0.
    weight_vector_to_choose = sample[np.random.randint(len(sample))]
    for i in sample:
        # print "CROSS VALIDATION %s" % i

        training_indices = [j for j in range(X.shape[0]) if j != i]
        W = SGD(X[training_indices,], Y[training_indices,], maxIter=maxIter, lmda=lmda, learningRate=learningRate)
        # print W
        if i == weight_vector_to_choose:
            final_W = W
        y_hat = np.sign(X[i,].dot(W))

        if y_hat == Y[i]:
            nCorrect += 1
        else:
            nIncorrect += 1

    return (nCorrect / (nCorrect + nIncorrect), final_W)

def generate_region_level_statistics(final_W):
    maxFeatures =  max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])
    num_images_per_trial = [data[i][0].shape[0] for i in range(data.shape[0])]
    max_num_images = max(num_images_per_trial)
    n_voxels_per_image = metadata['nvoxels'][0][0]
    np.savetxt('final_w.csv', final_W)
    W_matrix = final_W[0:maxFeatures].reshape(max_num_images, n_voxels_per_image)
    averaged_W = np.mean(W_matrix, axis=0)
    weights_grouped_by_region = []
    for roi in rois[0]:
        roi_name = roi[0][0]
        indices = roi[2][0] - 1
        weights_for_region = averaged_W[indices]
        weights_grouped_by_region.append({
            'name': roi_name,
            'total': np.sum(weights_for_region),
            'no_of_weights': len(weights_for_region),
            'weights': weights_for_region,
            'no_of_positive': np.sum(weights_for_region > 0),
            'no_of_negative': len(weights_for_region) - np.sum(weights_for_region > 0),
            'ratio': float(np.sum(weights_for_region > 0))/len(weights_for_region)
        })
    print("name,no_of_weights,no_of_positive_weights,no_of_negative_weights,percentage_of_positive_weights,sum_of_weights")
    for region in weights_grouped_by_region:
        print("%s,%s,%s,%s,%0.3f,%s"%(region['name'], region['no_of_weights'], region['no_of_positive'], region['no_of_negative'], region['ratio'], region['total']))

def main():
    maxFeatures =  max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])
    loss_function = "Hinge"
    loss_function = "Logistic"
    if loss_function == "Hinge":
        sgd_function = SgdHinge
        learning_rates = np.array([0.1, 0.01, 0.001, 0.0001])
        lambdas = np.array([1, 0.3, 0.1])
        best_learning_rate = 0.0001
        best_lambda = 1.0
        learning_rates = np.array([best_learning_rate])
        lambdas = np.array([best_lambda])
    else:
        learning_rates = np.array([0.1, 0.01, 0.001, 0.0001])
        lambdas = np.array([1, 0.3, 0.1])
        sgd_function = SgdLogistic
        best_learning_rate = 0.001
        best_lambda = 0.3
        # learning_rates = np.array([best_learning_rate])
        # lambdas = np.array([best_lambda])
    #Inputs
    X = np.zeros((ntrials, maxFeatures+1)).astype('float64')
    for i in range(data.shape[0]):
        f = data[i][0].flatten()
        X[i,:f.shape[0]] = f
        X[i,f.shape[0]]  = 1     #Bias

    #Outputs (+1 = Picture, -1 = Sentence)
    Y = np.ones(ntrials).astype('float64')
    Y[np.array([info[i]['firstStimulus'][0] != 'P' for i in range(ntrials)])] = -1

    #Randomly permute the data
    np.random.seed(1)      #Seed the random number generator to preserve the dev/test split
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation,]
    Y = Y[permutation,]

    best_accuracy = 0
    print("Comparing different models")
    for learning_rate in learning_rates:
        for lmda in lambdas:
            (training_accuracy, W) = crossValidation(X, Y, sgd_function, maxIter=100, lmda=lmda, learningRate=learning_rate)
            print ("Learning rate: %s, lambda: %s, Accuracy: %s"%(learning_rate, lmda, training_accuracy))
            if training_accuracy > best_accuracy:
                best_accuracy = training_accuracy
                best_learning_rate = learning_rate
                best_lambda = lmda

    print("\nBest model")
    print "Accuracy (%s Loss):\t%s learning_rate: %s lambda: %s" %(loss_function, best_accuracy, best_learning_rate, best_lambda)
    
    (test_accuracy, final_W) = crossValidation(X, Y, SgdHinge, maxIter=100, lmda=best_lambda, learningRate=best_learning_rate, sample=range(20,X.shape[0]))
    # final_W = np.genfromtxt('final_w.csv', dtype=float)
    print "Accuracy (%s Loss):\t%s" %(loss_function, test_accuracy)
    generate_region_level_statistics(final_W)

    return (X,Y)

if __name__ == "__main__":
    main()
