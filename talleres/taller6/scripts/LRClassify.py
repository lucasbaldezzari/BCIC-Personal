# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 20:50:41 2021

@author: Lucas

Testing a Linear Regression for SSVEPs classification
"""
import os
import numpy as np
import numpy.matlib as npm

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum
from utils import plotEEG
import fileAdmin as fa

#Transforming data for training
def getDataForClassifier(features, numClases, channel = 1):
    """
    Prepare the features set in order to fit and train the CNN model.
    
    Arguments:
        - features: Magnitud Spectrum Features or Complex Spectrum Features
    """
    
    print("Generating training data")
    # print("Original features shape: ", features.shape)
    featuresData = np.reshape(features, (features.shape[0], features.shape[1],features.shape[2],
                                         features.shape[3]*features.shape[4]))
    
    # print("featuresData shape: ", featuresData.shape)
    
    trainingData = featuresData[:, channel - 1, 0, :].T
    # print("Transpose trainData shape(1), ", trainingData.shape)
    
    # Reshaping the data into dim [classes*trials x features]    
    for target in range(1, featuresData.shape[2]):
        trainingData = np.vstack([trainingData, np.squeeze(featuresData[:, channel - 1, target, :]).T])
        
    # print("Transpose trainData shape(2), ", trainingData.shape)
    trainingData = np.reshape(trainingData, (trainingData.shape[0], trainingData.shape[1]))
    # print("Transpose trainData shape(3), ", trainingData.shape)
    
    epochsPerClass = featuresData.shape[3]
    featuresData = []
    
    classLabels = np.arange(numClases)
    labels = (npm.repmat(classLabels, epochsPerClass, 1).T).ravel()
    
    # # labels = to_categorical(labels).reshape(features.shape[0]*features.shape[2])     
    labels = to_categorical(labels)
    # print(labels.shape)
    
    return trainingData, labels


"""Let's starting"""
            
actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\talleres\\taller4\\scripts',"dataset")
# dataSet = sciio.loadmat(f"{path}/s{subject}.mat")

# path = "E:/reposBCICompetition/BCIC-Personal/taller4/scripts/dataset" #directorio donde estan los datos

subjects = np.arange(0,10)
subjectsNames = [f"s{subject}" for subject in np.arange(1,11)]

fm = 256.0
tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
muestraDescarte = 39
frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])

"""Loading the EEG data"""
rawEEGs = fa.loadData(path = path, filenames = subjectsNames)
#Original Shape (12 clsses, 8 channels, 1114 samples , 15 trials)

samples = rawEEGs["s2"]["eeg"].shape[2] #the are the same for all sobjecs and trials

#Filtering de EEG
PRE_PROCES_PARAMS = {
                'lfrec': 5.,
                'hfrec': 38.,
                'order': 4,
                'sampling_rate': fm,
                'window': 4,
                'shiftLen':4
                }

resolution = fm/samples

FFT_PARAMS = {
                'resolution': resolution,#0.2930,
                'start_frequency': 5.0,
                'end_frequency': 38.0,
                'sampling_rate': fm
                }

for subject in subjectsNames:
    eeg = rawEEGs[subject]["eeg"]
    eeg = eeg[:,:, muestraDescarte: ,:]
    eeg = eeg[:,:, :tiempoTotal ,:]
    rawEEGs[subject]["eeg"] = filterEEG(eeg,
                                        lfrec = PRE_PROCES_PARAMS["lfrec"],
                                        hfrec = PRE_PROCES_PARAMS["hfrec"],
                                        orden = 4, fm  = fm)

#Selecting the first 12 trials for training and validation
EEGS2 = rawEEGs["s2"]["eeg"][:, :, :, 0:12]
EEGS8 = rawEEGs["s8"]["eeg"][:, :, :, 0:12]

# plotEEG(EEGS8, sujeto = 8, trial = 1, blanco = 1,
#             fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

#Selecting the last 3 trials for testing
testEEGS2 = rawEEGs["s2"]["eeg"][:, :, :, 12:] #testing set
testEEGS8 = rawEEGs["s8"]["eeg"][:, :, :, 12:] #testing set

"""Getting features"""

#eeg data segmentation
EEGS2Segmented = segmentingEEG(EEGS2, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                               PRE_PROCES_PARAMS["sampling_rate"])
EEGS8Segmented = segmentingEEG(EEGS8, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                               PRE_PROCES_PARAMS["sampling_rate"])
testEEGS2Segmented = segmentingEEG(testEEGS2, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                   PRE_PROCES_PARAMS["sampling_rate"])
testEEGS8Segmented = segmentingEEG(testEEGS8, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                   PRE_PROCES_PARAMS["sampling_rate"])

#getting FFT spectrum for the training and validation set
#the shape is (number of features x channels x classes, trials)
EEGS2MagFFT = computeMagnitudSpectrum(EEGS2Segmented, FFT_PARAMS)
EEGS8MagFFT = computeMagnitudSpectrum(EEGS8Segmented, FFT_PARAMS)


plotSpectrum(EEGS2MagFFT, resolution, 12, 2, 1, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 2 - Channel 1 - Training set",
              folder = "figs")

plotSpectrum(EEGS8MagFFT, resolution, 12, 8, 1, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 2 - Channel 1 - Training set",
              folder = "figs")

#Average spectrum through channels
EEGS2MagFFTmean = np.mean(EEGS2MagFFT,axis = 1).reshape(144,1,12,12,1)
plotSpectrum(EEGS2MagFFTmean, resolution, 12, 8, 0, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 8 - Average through channels - Testing Set",
              folder = "figs")


EEGS8MagFFTmean = np.mean(EEGS2MagFFT,axis = 1).reshape(144,1,12,12,1)
plotSpectrum(EEGS8MagFFTmean, resolution, 12, 8, 0, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 2 - Average through channels - Testing Set",
              folder = "figs")

#getting FFT spectrum for the test set
testEEGS2MagFFT = computeMagnitudSpectrum(testEEGS2Segmented, FFT_PARAMS)
testEEGS8MagFFT = computeMagnitudSpectrum(testEEGS8Segmented, FFT_PARAMS)

testEEGS2MagFFTmean = np.mean(testEEGS2MagFFT,axis = 1).reshape(144,1,12,3,1)
testEEGS8MagFFTmean = np.mean(testEEGS8MagFFT,axis = 1).reshape(144,1,12,3,1)

plotSpectrum(testEEGS2MagFFT, resolution, 12, 8, 7, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 2 - Channel 7 - Testing Set",
              folder = "figs")

plotSpectrum(testEEGS8MagFFT, resolution, 12, 8, 7, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 8 - Channel 7 - Testing Set",
              folder = "figs")

trainSetEEGS2, train_labelsEEGS2 = getDataForClassifier(EEGS2MagFFT, numClases = 12)
trainSetEEGS8, train_labelsEEGS8 = getDataForClassifier(EEGS8MagFFT, numClases = 12)

testSetEEGS2, test_labelsEEGS2 = getDataForClassifier(testEEGS2MagFFT, numClases = 12)
testSetEEGS8, test_labelsEEGS2 = getDataForClassifier(testEEGS8MagFFT, numClases = 12)

avgTrainSetEEGS2, avgTrain_labelsEEGS2 = getDataForClassifier(EEGS2MagFFTmean, numClases = EEGS2MagFFTmean.shape[2])
avgTrainSetEEGS8, avgTrain_labelsEEGS8 = getDataForClassifier(EEGS8MagFFTmean, numClases = EEGS8MagFFTmean.shape[2])

avgTestSetEEGS2, avgTest_labelsEEGS2 = getDataForClassifier(testEEGS2MagFFTmean, numClases = testEEGS2MagFFTmean.shape[2])
avgTestSetEEGS8, avgTest_labelsEEGS8 = getDataForClassifier(testEEGS8MagFFTmean, numClases = testEEGS8MagFFTmean.shape[2])

# Cross validation
partitionsGenerator =  StratifiedKFoldgenerador_particiones = KFold(n_splits=5, random_state=0, shuffle=True)

trainSetEEG8Partitions = list(partitionsGenerator.split(avgTrainSetEEGS8, avgTrain_labelsEEGS8))

partitionsGenerator =  StratifiedKFoldgenerador_particiones = KFold(n_splits=5, random_state=0, shuffle=True)
trainSetEEG2Partitions = list(partitionsGenerator.split(avgTrainSetEEGS2, avgTrain_labelsEEGS2))

generalResults = []

model = LogisticRegression()

fold = 1
channel = 7 #use the channel 7 to training the model
clase = 2

modelsS8 = dict()
modelNumber = 1
for clase in np.arange(0,12):
    generalResults = []
    fold = 1
    model = LogisticRegression() #inicializo modelo
    accu = 0
    for train_ind, test_ind in tqdm(trainSetEEG8Partitions): 
    
        model.fit(avgTrainSetEEGS8[train_ind, :], avgTrain_labelsEEGS8[train_ind, clase]) #entreno modelo
    
        pred = model.predict(avgTrainSetEEGS8[test_ind, :])
        if accuracy_score(avgTrain_labelsEEGS8[test_ind,:][:,clase], pred) > accu:
            
            accu = accuracy_score(avgTrain_labelsEEGS8[test_ind,:][:,clase], pred)
            modelsS8[f"model{modelNumber}"] = (model, accu)

    modelNumber += 1

clase = 1
#predict the first three data corresponding to class 1 using the model for predict class 1
modelsS8["model1"][0].predict(avgTestSetEEGS8[(clase-1)*3:(clase-1) *3 +3 , :])

for clase in np.arange(1,13):
    print(modelsS8[f"model{clase}"][0].predict(avgTestSetEEGS8[(clase-1)*3:(clase-1) *3 +3, :]))

clase = 12
#predict the first three data corresponding to class 2 using the model for predict class 3
modelsS8["model12"][0].predict(avgTestSetEEGS8[(clase-1)*3:(clase-1) *3 +3 , :])

#predict one trial for class 1 using the model to predit class 1
modelsS8["model1"][0].predict(avgTestSetEEGS8[0,:].reshape(1, -1))

modelsS2 = dict()
modelNumber = 1
for clase in np.arange(0,12):
    generalResults = []
    fold = 1
    model = LogisticRegression() #inicializo modelo
    accu = 0
    for train_ind, test_ind in tqdm(trainSetEEG2Partitions): 
    
        model.fit(avgTrainSetEEGS2[train_ind, :], avgTrain_labelsEEGS2[train_ind, clase]) #entreno modelo
    
        pred = model.predict(avgTrainSetEEGS2[test_ind, :])
        if accuracy_score(avgTrain_labelsEEGS2[test_ind,:][:,clase], pred) > accu:
            
            accu = accuracy_score(avgTrain_labelsEEGS2[test_ind,:][:,clase], pred)
            modelsS2[f"model{modelNumber}"] = (model, accu)

    modelNumber += 1

clase = 12
#predict the first three data corresponding to class 1 using the model for predict class 1
modelsS2[f"model{clase}"][0].predict(avgTestSetEEGS2[(clase-1)*3:(clase-1) *3 +3, :])

for clase in np.arange(1,13):
    print(modelsS2[f"model{clase}"][0].predict(avgTestSetEEGS2[(clase-1)*3:(clase-1) *3 +3, :]))

clase = 7
#predict the first three data corresponding to class 2 using the model for predict class 1
modelsS2["model7"][0].predict(avgTestSetEEGS2[(clase-1)*3:(clase-1) *3 +3 , :])

#predict one trial for class 1 using the model to predit class 1
modelsS2["model1"][0].predict(avgTestSetEEGS2[0,:].reshape(1, -1))

"""Make the same process for all the subjects"""
matrixTrainResults = np.zeros((len(subjectsNames),len(frecStimulus)))
modelsForSubject = dict()
testSetForSubject = dict()
for i, subject in enumerate(tqdm(subjectsNames)):
    
    #Selecting the first 12 trials for training and validation
    trainSet = rawEEGs[subject]["eeg"][:, :, :, 0:12]
    
    #Selecting the last 3 trials for testing
    testSet = rawEEGs[subject]["eeg"][:, :, :, 12:] #testing set
        
    #eeg data segmentation
    segmented_trainSet = segmentingEEG(trainSet, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                         PRE_PROCES_PARAMS["sampling_rate"])

    segmented_testingSet = segmentingEEG(testSet, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                         PRE_PROCES_PARAMS["sampling_rate"])
    
    #getting FFT spectrum for the training and validation set
    #the shape is (number of features x channels x classes, trials)
    TRAIN_SET = computeMagnitudSpectrum(segmented_trainSet, FFT_PARAMS)
    TEST_SET = computeMagnitudSpectrum(segmented_testingSet, FFT_PARAMS)
    
    channel = 1
    
    plotSpectrum(TRAIN_SET, resolution, 12, subject, channel, frecStimulus,
                  startFrecGraph = FFT_PARAMS['start_frequency'],
                  save = False, title = f"Average maggnitud features - {subject} - Channel {channel} - Train set",
                  folder = "figs")
    
    #Average spectrum through channels
    #Another option is select the best channel and use it for train the classifier
    featuresNumber = TRAIN_SET.shape[0]
    channelNumber = 1
    classes = TRAIN_SET.shape[2]
    trials = TRAIN_SET.shape[3]
    avg_TRAIN_SET = np.mean(TRAIN_SET,axis = 1).reshape(featuresNumber,channelNumber,classes,trials,1)
    
    featuresNumber = TEST_SET.shape[0]
    channelNumber = 1
    classes = TEST_SET.shape[2]
    trials = TEST_SET.shape[3]
    avg_TEST_SET = np.mean(TEST_SET,axis = 1).reshape(featuresNumber,channelNumber,classes,trials,1)
    
    avgTrainSet, avgTrain_labels = getDataForClassifier(avg_TRAIN_SET, numClases = avg_TRAIN_SET.shape[2])

    avgTestSet, avgTest_labels = getDataForClassifier(avg_TEST_SET, numClases = avg_TEST_SET.shape[2])
    
    testSetForSubject[subject] = avgTestSet
    # Cross validation
    partitionsGenerator =  StratifiedKFoldgenerador_particiones = KFold(n_splits=5, random_state=0, shuffle=True)

    trainSetPartitions = list(partitionsGenerator.split(avgTrainSet, avgTrain_labels))
    testSetPartitions = list(partitionsGenerator.split(avgTestSet, avgTest_labels))

    #training the model
    model = LogisticRegression()
    
    models = dict()
    for j, clase in enumerate(np.arange(0,12)):
        generalResults = []
        model = LogisticRegression() #inicializo modelo
        accu = 0
        for train_ind, test_ind in trainSetPartitions: 
        
            model.fit(avgTrainSet[train_ind, :], avgTrain_labels[train_ind, clase]) #entreno modelo
        
            pred = model.predict(avgTrainSet[test_ind, :])
            if accuracy_score(avgTrain_labels[test_ind,:][:,clase], pred) > accu:
                
                accu = accuracy_score(avgTrain_labels[test_ind,:][:,clase], pred)
                models[f"{frecStimulus[clase]}"] = (model, accu)
                matrixTrainResults[i][j] = accu
                
    modelsForSubject[f"{subject}"] = models
    

plt.figure(figsize=(15,10))
plt.imshow(matrixTrainResults)
plt.xlabel("Accuracy predicion for Clasess")
plt.xticks(np.arange(len(frecStimulus)), frecStimulus)
plt.ylabel("Subjects")
plt.yticks(np.arange(len(subjectsNames)), subjectsNames)
plt.colorbar();

for i in range(matrixTrainResults.shape[0]):
    for j in range(matrixTrainResults.shape[1]):
        plt.text(j, i, "{:.2f}".format(matrixTrainResults[i, j]), va='center', ha='center')
plt.show()

clase = 1 #stimulus 9.25
#predict the first three data corresponding to class 1 using the model for predict class 1 for subject 1
modelsForSubject["s8"][f"{frecStimulus[clase-1]}"][0].predict(testSetForSubject["s8"][(clase-1)*3:(clase-1) *3 +3, :])

clase = 5 #stimulus 11.75
#predict the first three data corresponding to class 5 using the model for predict class 5 for subject 6
modelsForSubject["s6"][f"{frecStimulus[clase-1]}"][0].predict(testSetForSubject["s6"][(clase-1)*3:(clase-1) *3 +3, :])

clase = 3 #stimulus 13.25
#predict the first three data corresponding to class 12 using the model for predict class 12 for subject 9
modelsForSubject["s9"][f"{frecStimulus[clase-1]}"][0].predict(testSetForSubject["s9"][(clase-1)*3:(clase-1) *3 +3, :])

clase = 12 #stimulus 14.75
#predict the first three data corresponding to class 12 using the model for predict class 12 for subject 2
modelsForSubject["s2"][f"{frecStimulus[clase-1]}"][0].predict(testSetForSubject["s2"][(clase-1)*3:(clase-1) *3 +3, :])
