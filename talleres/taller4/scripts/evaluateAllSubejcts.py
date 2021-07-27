# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:08:18 2021

@author: Lucas
"""

import sys
import os

import warnings
import numpy as np
import numpy.matlib as npm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold

import tensorflow as tf

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

from brainflow.data_filter import DataFilter

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import barPlotSubjects

from CNNTrainingModule import *
from CNNClassify import *

def trainForAllSubject():
    
    import fileAdmin as fa
                
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"dataset")
    
    subjects = [1,2,3,4,5,6,7,8,9,10]
    
    fm = 256.0
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
    
    """
    **********************************************************************
    First step: Loading and plotting the EEG
    **********************************************************************
    """
    for subject in subjects:

        rawEEG = fa.loadData(path = path, subjects = subjects)[f"s{subject}"]["eeg"]
        
        #Selecting the first 12 trials
        rawEEG = rawEEG[:, :, :, 0:12]
        
        samples = rawEEG.shape[2]
        resolution = fm/samples
        
        rawEEG = rawEEG[:,:, muestraDescarte: ,:]
        rawEEG = rawEEG[:,:, :tiempoTotal ,:]
        
        PRE_PROCES_PARAMS = {
                        'lfrec': 5.,
                        'hfrec': 38.,
                        'order': 4,
                        'sampling_rate': fm,
                        'window': 4,
                        'shiftLen':4
                        }
        
        CNN_PARAMS = {
                        'batch_size': 64,
                        'epochs': 50,
                        'droprate': 0.25,
                        'learning_rate': 0.001,
                        'lr_decay': 0.0,
                        'l2_lambda': 0.0001,
                        'momentum': 0.9,
                        'kernel_f': 10,
                        'n_ch': 8,
                        'num_classes': 12}
        
        FFT_PARAMS = {
                        'resolution': resolution,#0.2930,
                        'start_frequency': 5.0,
                        'end_frequency': 38.0,
                        'sampling_rate': fm
                        }
 
        #filtro la señal entre los 5hz y los 80hz
        eegfiltrado = filterEEG(rawEEG,
                                lfrec = PRE_PROCES_PARAMS["lfrec"],
                                hfrec = PRE_PROCES_PARAMS["hfrec"], orden = 4, fm  = 256.0)
        
        """
        **********************************************************************
        Second step: Create the CNN
        **********************************************************************
        """
        
        #Make a CNNTrainingModule object in order to use the data's Magnitude Features
        magnitudCNN = CNNTrainingModule(rawEEG = rawEEG, subject = subject,
                                        PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                        FFT_PARAMS = FFT_PARAMS,
                                        CNN_PARAMS = CNN_PARAMS,
                                        modelName = f"CNN_UsingMagnitudFeatures_Subject{subject}")
        
        """
        **********************************************************************
        Third step: Compute and get the Magnitud Spectrum Features
        **********************************************************************
        """
        #Computing and getting the magnitude Spectrum Features
        magnitudFeatures = magnitudCNN.computeMSF()
        
        # Get the training and testing data for CNN using Magnitud Spectrum Features
        trainingData_MSF, labels_MSF = magnitudCNN.getDataForTraining(magnitudFeatures)
        
        #inputshape [Number of Channels x Number of Features x 1]
        inputshape = np.array([trainingData_MSF.shape[1], trainingData_MSF.shape[2], trainingData_MSF.shape[3]])
        
        
        """
        **********************************************************************
        Fourth step: Create the CNN model
        **********************************************************************
        """
        magnitudCNN.createModel(inputshape)
        
        """
        **********************************************************************
          Fifth step: Trainn the CNN
        **********************************************************************
        """
        
        accu_CNN_using_MSF = magnitudCNN.trainCNN(trainingData_MSF, labels_MSF, nFolds = 10)
        
        #saving the model
        magnitudCNN.saveCNNModel()

        """Make a CNNTrainingModule object in order to use the data's Magnitude and data's Complex Features"""
        complexCNN = CNNTrainingModule(rawEEG = rawEEG, subject = subject,
                                        PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                        FFT_PARAMS = FFT_PARAMS,
                                        CNN_PARAMS = CNN_PARAMS,
                                        modelName = f"CNN_UsingComplexFeatures_Subject{subject}")
        
        complexFeatures = complexCNN.computeCSF()
        
        # Training and testing CNN suing Complex Spectrum Features
        trainingData_CSF, labels_CSF = complexCNN.getDataForTraining(complexFeatures)
        
        #inputshape [Number of Channels x Number of Features x 1]
        inputshape = np.array([trainingData_CSF.shape[1], trainingData_CSF.shape[2], trainingData_CSF.shape[3]])
        
        # Create the CNN model
        complexCNN.createModel(inputshape)
        
        accu_CNN_using_CSF = complexCNN.trainCNN(trainingData_CSF, labels_CSF, nFolds = 10)
        
        complexCNN.saveCNNModel()

def evaluateAllSubejcts():
    
    import fileAdmin as fa

    actualFolder = os.getcwd() #directorio donde estamos actualmente. Debe contener el directorio dataset
    baseFolder = actualFolder
    path = os.path.join(actualFolder,"dataset")
    
    subjects = [1,2,3,4,5,6,7,8,9,10]
        
    fm = 256.0
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25,
                              9.75, 11.75, 13.75,
                              10.25, 12.25, 14.25,
                              10.75, 12.75, 14.75])
    
    totalMagnitudAccuracies = dict()
    totalComplexAccuracies = dict()
    magnitudAccuList = []
    complexAccuList = []
    
    totalMagnitudAccuracies = dict()
    
    for subject in subjects:
    
        """
        **********************************************************************
        Loading and plotting the EEG
        
        IMPORTANT: In real time BCI, the "loading data" will be replaced
        for real data coming from the OpenBCI board
        **********************************************************************
        """
        
        # MagnitudAccuraciesList[f"subject{subject}"] = list()
        # ComplexAccuraciesList[f"subject{subject}"] = list()
        
        totalMagnitudAccuracies[f"subject{subject}"] = {"magnitudAccu":list(),
                                                        "complexAccu":list()}
        
        actualFolder = baseFolder
        path = os.path.join(actualFolder,"dataset")
        rawEEG = fa.loadData(path = path, subjects = subjects)[f"s{subject}"]["eeg"]
        
        #selec the last 3 trials
        rawEEG = rawEEG[:, :, :, 10:]
        samples = rawEEG.shape[2]
        resolution = fm/samples
        
        PRE_PROCES_PARAMS = {
                        'lfrec': 3.,
                        'hfrec': 36.,
                        'order': 4,
                        'sampling_rate': fm,
                        'window': 4,
                        'shiftLen':4
                        }
        
        
        FFT_PARAMS = {
                        'resolution': resolution,#0.2930,
                        'start_frequency': 5.0,
                        'end_frequency': 38.0,
                        'sampling_rate': fm
                        }
        
        rawEEG = rawEEG[:,:, muestraDescarte: ,:]
        rawEEG = rawEEG[:,:, :tiempoTotal ,:]
        
        """
        **********************************************************************
        First step: Create CNNClassify object in order to load a trained model
        and classify new data
        **********************************************************************
        """
        
        # create an CNNClassify object in order to work with magnitud features
        magnitudCNNClassifier = CNNClassify(modelFile = f"CNN_UsingMagnitudFeatures_Subject{subject}",
                                    weightFile = f"bestWeightss_CNN_UsingMagnitudFeatures_Subject{subject}",
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    classiName = f"CNN_Classifier",
                                    frecStimulus = frecStimulus.tolist())
        
        complexCNNClassifier = CNNClassify(modelFile = f"CNN_UsingComplexFeatures_Subject{subject}",
                                    weightFile = f"bestWeightss_CNN_UsingComplexFeatures_Subject{subject}",
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    classiName = f"CNN_Classifier",
                                    frecStimulus = frecStimulus.tolist())
    
        stimuli = [1,2,3,4,5,6,7,8,9,10,11,12]
        trials = [1,2,3]
        
        for stimulus in stimuli:
            print("*****************************")
            print(f"Stimulus {frecStimulus[stimulus-1]}")
            print("*****************************")
            
            magnitudTotalScore = 0 
            complexTotalScore = 0
            
            for trial in trials:
                # stimulus = 1 #slected stimulus for classification
                # trial = 1 #selected trial
                 
                #get the selected trial and stimulus from rawEEG
                data = rawEEG[stimulus-1,:,:,trial-1].reshape(1, rawEEG.shape[1],rawEEG.shape[2],1)
                path = os.path.join(actualFolder,"models")
    
                """
                **********************************************************************
                Second step: Create CNNClassify object in order to load a trained model
                and classify new data
                **********************************************************************
                """
                
                # get the features for my data
                magnitudFeatures = magnitudCNNClassifier.computeMSF(data)
                complexFeatures = complexCNNClassifier.computeCSF(data)
                
                # Prepare my data for classification. This is important, the input data for classification
                # must be the same shape the CNN was trained.
                magnitudDataForClassification = magnitudCNNClassifier.getDataForClassification(magnitudFeatures)
                complexDataForClassification = complexCNNClassifier.getDataForClassification(complexFeatures)
                
                # Get a classification. The classifyEEGSignal() method give us a stimulus
                magnitudClassification = magnitudCNNClassifier.classifyEEGSignal(magnitudDataForClassification)
                complexClassification = complexCNNClassifier.classifyEEGSignal(complexDataForClassification)

                if magnitudClassification == frecStimulus[stimulus-1]:
                    magnitudTotalScore += 1
                    
                if complexClassification == frecStimulus[stimulus-1]:
                    complexTotalScore += 1
                    
            # totalMagnitudAccuracies[f"subject{subject}"].append(magnitudTotalScore/len(trials))
            
            totalMagnitudAccuracies[f"subject{subject}"]["magnitudAccu"].append(magnitudTotalScore/len(trials))
            totalMagnitudAccuracies[f"subject{subject}"]["complexAccu"].append(complexTotalScore/len(trials))
            
    return totalMagnitudAccuracies

#Call this function in order to train the classifiers for all subjects
# trainForAllSubject()

#Get the total accuracies
totalMagnitudAccuracies = evaluateAllSubejcts()

#Average accuracies for subject 8
print(totalMagnitudAccuracies[f"subject{8-1}"])

#Average complex accuracies for subject 8
print(totalMagnitudAccuracies[f"subject{8-1}"]["complexAccu"])

#Average accuracy for stimulus 2 subject 8
print(totalMagnitudAccuracies[f"subject{8-1}"]["complexAccu"][2-1]*100)

mean = list()
svd = list()

for subject in totalMagnitudAccuracies:
    mean.append(np.mean(totalMagnitudAccuracies[subject]["magnitudAccu"])*100)

stimuli = np.array([9.25, 11.25, 13.25,
                              9.75, 11.75, 13.75,
                              10.25, 12.25, 14.25,
                              10.75, 12.75, 14.75])

title = f"Average accuracy for stimuli for all subjects - Magnitud Features"
barPlotSubjects(mean, 0,
                    etiquetas = ["s1", "s2", "s3", "s4", "s5", "s6","s7","s8","s9","s10"],
                    savePlots = True,
                    title = title)

stimulus = 12
meanMagAccuStimulus = list()    

for subject in totalMagnitudAccuracies:
    meanMagAccuStimulus.append(np.mean(totalMagnitudAccuracies[subject]["complexAccu"][stimulus-1])*100)
    
title = f"Average accuracy for stimulus {stimuli[stimulus-1]} for all subjects - Magnitud Features"
barPlotSubjects(meanMagAccuStimulus, 0,
                    etiquetas = ["s1", "s2", "s3", "s4", "s5", "s6","s7","s8","s9","s10"],
                    savePlots = True,
                    title = title)