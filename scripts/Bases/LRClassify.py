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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum
from utils import plotEEG
import fileAdmin as fa

#Transforming data for training
def getDataForClassifier(features, numClases, channel = 1):
    """Prepare the features set in order to fit and train the CNN model.
        
    Arguments:
        - features: Magnitud Spectrum Features or Complex Spectrum Features
        with shape [number of features x channels x classes x trials x 1]
        
    Return:
        - trainingData: Data to fit the LR model or to classify using a LR model.
        With shape [trials*clases x number of features]
        - Labels: labels for training the LR model
    """
    
    print("Generating training data")
    # print("Original features shape: ", features.shape)
    featuresData = np.reshape(features, (features.shape[0], features.shape[1],features.shape[2],
                                         features.shape[3]*features.shape[4]))
    
    # print("featuresData shape: ", featuresData.shape)
    
    dataForClassify = featuresData[:, channel - 1, 0, :].T
    # print("Transpose dataForClassify shape(1), ", dataForClassify.shape)
    
    # Reshaping the data into dim [classes*trials x features]    
    for target in range(1, featuresData.shape[2]):
        dataForClassify = np.vstack([dataForClassify, np.squeeze(featuresData[:, channel - 1, target, :]).T])
        
    # print("Transpose dataForClassify shape(2), ", dataForClassify.shape)
    dataForClassify = np.reshape(dataForClassify, (dataForClassify.shape[0], dataForClassify.shape[1]))
    # print("Transpose dataForClassify shape(3), ", dataForClassify.shape)
    
    epochsPerClass = featuresData.shape[3]
    featuresData = []
    
    classLabels = np.arange(numClases)
    labels = (npm.repmat(classLabels, epochsPerClass, 1).T).ravel()
      
    labels = to_categorical(labels)
    # print(labels.shape)
    
    return dataForClassify, labels


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

#Filtering the EEG
for subject in subjectsNames:
    eeg = rawEEGs[subject]["eeg"]
    eeg = eeg[:,:, muestraDescarte: ,:]
    eeg = eeg[:,:, :tiempoTotal ,:]
    rawEEGs[subject]["eeg"] = filterEEG(eeg,
                                        lfrec = PRE_PROCES_PARAMS["lfrec"],
                                        hfrec = PRE_PROCES_PARAMS["hfrec"],
                                        orden = 4, fm  = fm)

#Selecting the first 12 trials for training and validation
EEGS8 = rawEEGs["s8"]["eeg"][:, :, :, 0:12]

#Selecting the last 3 trials for testing
testEEGS8 = rawEEGs["s8"]["eeg"][:, :, :, 12:] #testing set

components = 4
clases = EEGS8.shape[0]
channels = EEGS8.shape[1]
samples = EEGS8.shape[2]
trials = EEGS8.shape[3]

eegS8_PC = np.zeros((clases, components, samples, trials))

for clase in np.arange(0,12):
    pca = PCA(n_components = components)

    data = EEGS8[clase,:,:,:].reshape(channels, samples*trials).swapaxes(0,1)
    pca.fit(data)
    eegS8_PC[clase] = pca.transform(data).swapaxes(0,1).reshape(components,samples,trials)
    
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# plotEEG(EEGS8, sujeto = 8, trial = 1, blanco = 1,
#             fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")



"""Getting features"""

#eeg data segmentation
EEGS8Segmented = segmentingEEG(eegS8_PC, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                               PRE_PROCES_PARAMS["sampling_rate"])

testEEGS8Segmented = segmentingEEG(testEEGS8, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                   PRE_PROCES_PARAMS["sampling_rate"])

#getting FFT spectrum for the training and validation set
#the shape is (number of features x channels x classes, trials)
EEGS8MagFFT = computeMagnitudSpectrum(EEGS8Segmented, FFT_PARAMS)

canal = 1 #first component
sujeto = 8
clases = 12
plotSpectrum(EEGS8MagFFT, resolution, clases, sujeto, canal-1, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 8 - Channel 1 - Training set",
              folder = "figs")

#Average spectrum through channels
features = EEGS8MagFFT.shape[0]
nchannels = 1
clases = EEGS8MagFFT.shape[2]
trials = EEGS8MagFFT.shape[3]
avg_EEGS8MagFFT = np.mean(EEGS8MagFFT,axis = 1).reshape(features,nchannels,clases,trials,1)

plotSpectrum(avg_EEGS8MagFFT, resolution, 12, 1, 0, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 8 - Average through channels - Testing Set",
              folder = "figs")

#getting FFT spectrum for the test set
testEEGS8MagFFT = computeMagnitudSpectrum(testEEGS8Segmented, FFT_PARAMS)

features = testEEGS8MagFFT.shape[0]
nchannels = 1
clases = testEEGS8MagFFT.shape[2]
trials = testEEGS8MagFFT.shape[3]

avg_testEEGS8MagFFT = np.mean(testEEGS8MagFFT,axis = 1).reshape(features,nchannels,clases,trials,1)

plotSpectrum(testEEGS8MagFFT, resolution, 12, 8, 7, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "Maagnitud features for Subject 8 - Channel 7 - Testing Set",
              folder = "figs")

trainSetEEGS8, train_labelsEEGS8 = getDataForClassifier(EEGS8MagFFT, numClases = 12)

avgTrainSetEEGS8, avgTrain_labelsEEGS8 = getDataForClassifier(avg_EEGS8MagFFT, numClases = avg_EEGS8MagFFT.shape[2])


# Cross validation
partitionsGenerator =  StratifiedKFoldgenerador_particiones = KFold(n_splits=4, random_state=0, shuffle=True)

trainSetEEG8Partitions = list(partitionsGenerator.split(avgTrainSetEEGS8, avgTrain_labelsEEGS8))

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
    for train_ind, test_ind in trainSetEEG8Partitions: 
    
        model.fit(avgTrainSetEEGS8[train_ind, :], avgTrain_labelsEEGS8[train_ind, clase]) #entreno modelo
    
        pred = model.predict(avgTrainSetEEGS8[test_ind, :])
        if accuracy_score(avgTrain_labelsEEGS8[test_ind,:][:,clase], pred) > accu:
            
            accu = accuracy_score(avgTrain_labelsEEGS8[test_ind,:][:,clase], pred)
            modelsS8[f"model{modelNumber}"] = (model, accu)

    modelNumber += 1

#predict the first three data corresponding to class 1 using the model for predict class 1
clase = 1
numberFeatures = avg_testEEGS8MagFFT.shape[0]
trials = avg_testEEGS8MagFFT.shape[3]
data = avg_testEEGS8MagFFT[:, :, clase - 1, :, :].reshape(numberFeatures,trials).swapaxes(0,1)
modelsS8["model1"][0].predict(data)

#predict the first three data corresponding to class 2 using the model for predict class 1
clase = 2
numberFeatures = avg_testEEGS8MagFFT.shape[0]
trials = avg_testEEGS8MagFFT.shape[3]
data = avg_testEEGS8MagFFT[:, :, clase - 1, :, :].reshape(numberFeatures,trials).swapaxes(0,1)
modelsS8["model1"][0].predict(data)

#predict using trial 1 for class 1 using the model to predit class 1
clase = 1
trial = 1
data = avg_testEEGS8MagFFT[:, 0, clase -1, trial - 1, 0].reshape(1, -1)
modelsS8["model1"][0].predict(data)


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
    
    # testSetForSubject[subject] = avgTestSet
    testSetForSubject[subject] = avg_TEST_SET
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
# modelsForSubject["s8"][f"{frecStimulus[clase-1]}"][0].predict(testSetForSubject["s8"][(clase-1)*3:(clase-1) *3 +3, :])
numberFeatures = testSetForSubject["s8"].shape[0]
trials = testSetForSubject["s8"].shape[3]
data = testSetForSubject["s8"][:, :, clase - 1, :, :].reshape(numberFeatures,trials).swapaxes(0,1)
modelsForSubject["s8"][f"{frecStimulus[clase-1]}"][0].predict(data)

#Predicting the first trial for clase 1 - subject 8
data = testSetForSubject["s8"][:, :, clase - 1, 0, :].reshape(1,numberFeatures)
modelsForSubject["s8"][f"{frecStimulus[clase-1]}"][0].predict(data)

clase = 5 #stimulus 11.75
#predict the first three data corresponding to class 5 using the model for predict class 5 for subject 6
numberFeatures = testSetForSubject["s6"].shape[0]
trials = testSetForSubject["s6"].shape[3]
data = testSetForSubject["s6"][:, :, clase - 1, :, :].reshape(numberFeatures,trials).swapaxes(0,1)
modelsForSubject["s6"][f"{frecStimulus[clase-1]}"][0].predict(data)

clase = 3 #stimulus 13.25
#predict the first three data corresponding to class 12 using the model for predict class 12 for subject 9
numberFeatures = testSetForSubject["s9"].shape[0]
trials = testSetForSubject["s9"].shape[3]
data = testSetForSubject["s9"][:, :, clase - 1, :, :].reshape(numberFeatures,trials).swapaxes(0,1)
modelsForSubject["s9"][f"{frecStimulus[clase-1]}"][0].predict(data)

clase = 12 #stimulus 14.75
#predict the first three data corresponding to class 12 using the model for predict class 12 for subject 2
numberFeatures = testSetForSubject["s2"].shape[0]
trials = testSetForSubject["s2"].shape[3]
data = testSetForSubject["s2"][:, :, clase - 1, :, :].reshape(numberFeatures,trials).swapaxes(0,1)
modelsForSubject["s2"][f"{frecStimulus[clase-1]}"][0].predict(data)

"""Make the same process but now using a PCA for selecting channels"""
matrixTrainResults = np.zeros((len(subjectsNames),len(frecStimulus)))
modelsForSubject = dict()
testSetForSubject = dict()
for i, subject in enumerate(tqdm(subjectsNames)):
    
    #Selecting the first 12 trials for training and validation
    trainSet = rawEEGs[subject]["eeg"][:, :, :, 0:12]
    
    #Selecting the last 3 trials for testing
    testSet = rawEEGs[subject]["eeg"][:, :, :, 12:] #testing set
    
    components = 4
    clases = trainSet.shape[0]
    channels = trainSet.shape[1]
    samples = trainSet.shape[2]
    trials = trainSet.shape[3]
    
    #Implementing a PCA in order to select 4 channels
    eeg_PC = np.zeros((clases, components, samples, trials))
    
    for clase in np.arange(0,12):
        pca = PCA(n_components = components)
    
        data = trainSet[clase,:,:,:].reshape(channels, samples*trials).swapaxes(0,1)
        pca.fit(data)
        eeg_PC[clase] = pca.transform(data).swapaxes(0,1).reshape(components,samples,trials)
        
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
    
    # testSetForSubject[subject] = avgTestSet
    testSetForSubject[subject] = avg_TEST_SET
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

