# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 19:08:42 2021

@author: Lucas BALDEZZARI

************ VERSIÓN: SCT-01-RevB ************

- Se agregan los métodos getMagnitudFeatureVector y getComplexFeatureVector

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
from utils import plotSpectrum, plotOneSpectrum

class CNNClassify():
    
    def __init__(self, modelFile, weightFile, frecStimulus,
                 PRE_PROCES_PARAMS, FFT_PARAMS, classiName = ""):
        """
        Some important variables configuration and initialization in order to implement a CNN 
        model for CLASSIFICATION.
        
        Args:
            - modelFile: File's name to load the pre trained model.
            - weightFile: File's name to load the pre model's weights
            - PRE_PROCES_PARAMS: The params used in order to pre process the raw EEG.
            - FFT_PARAMS: The params used in order to compute the FFT
            - CNN_PARAMS: The params used for the CNN model.
        """
        
        # load model from JSON file
        with open(f"models/{modelFile}.json", "r") as json_file:
            loadedModel = json_file.read()
        
            self.loadedModel = model_from_json(loadedModel)
            
        self.loadedModel.load_weights(f"models/{weightFile}.h5")
        self.loadedModel.make_predict_function()
        
        self.CSF = np.array([]) #Momplex Spectrum Features
        self.MSF = np.array([]) #Magnitud Spectrum Features
        
        self.classiName = classiName #Classfier object name
        
        self.frecStimulusList = frecStimulus
        
        #Setting variables for EEG processing.
        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS

        
    def classifyEEGSignal(self, dataForClassification):
        """
        Method used to classify new data.
        
        Args:
            - dataForClassification: Data for classification. The shape must be
            []
        """
        self.preds = self.loadedModel.predict(dataForClassification)
        
        return self.frecStimulusList[np.argmax(self.preds[0])]
        # return np.argmax(self.preds[0]) #máximo índice
    
    def computeMSF(self, rawEEG):
            """
            Compute the FFT over segmented EEG data.
            
            Argument:
                - None. This method use variables from the own class
            
            Return:
                - The Magnitud Spectrum Feature (MSF).
            """
            
            #eeg data filtering
            filteredEEG = filterEEG(rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                    self.PRE_PROCES_PARAMS["hfrec"],
                                    self.PRE_PROCES_PARAMS["order"],
                                    self.PRE_PROCES_PARAMS["sampling_rate"])
            
            #eeg data segmentation
            eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                         self.PRE_PROCES_PARAMS["shiftLen"],
                                         self.PRE_PROCES_PARAMS["sampling_rate"])
            
            self.MSF = computeMagnitudSpectrum(eegSegmented, self.FFT_PARAMS)
            
            return self.MSF
        
    def computeCSF(self, rawEEG):
            """
            Compute the FFT over segmented EEG data.
            
            Argument:
                - None. This method use variables from the own class
            
            Return:
                - The Complex Spectrum Feature (CSF) and the MSF in the same matrix
            """
            
            #eeg data filtering
            filteredEEG = filterEEG(rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                    self.PRE_PROCES_PARAMS["hfrec"],
                                    self.PRE_PROCES_PARAMS["order"],
                                    self.PRE_PROCES_PARAMS["sampling_rate"])
            
            #eeg data segmentation
            #shape: [targets, canales, trials, segments, duration]
            eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                         self.PRE_PROCES_PARAMS["shiftLen"],
                                         self.PRE_PROCES_PARAMS["sampling_rate"])
            
            #shape [2*n_fc, num_channels, num_classes, num_trials, number_of_segments]
            self.CSF = computeComplexSpectrum(eegSegmented, self.FFT_PARAMS)
            
            return self.CSF

    def getDataForClassification(self, features):
        """
        Prepare the features set in order to fit the CNN model and get a classification.
        
        Arguments:
            - features: Magnitud Spectrum Features or Complex Spectrum Features with shape
            
            [targets, channels, trials, segments, samples].
        
        """
        
        # print("Generating data for classification")
        # print("Original features shape: ", features.shape)
        featuresData = np.reshape(features, (features.shape[0], features.shape[1],features.shape[2],
                                             features.shape[3]*features.shape[4]))
        
        # print("featuresData shape: ", featuresData.shape)
        
        dataForClassification = featuresData[:, :, 0, :].T
        # print("Transpose trainData shape(1), ", dataForClassification.shape)
        
        #Reshaping the data into dim [classes*trials x channels x features]
        for target in range(1, featuresData.shape[2]):
            dataForClassification = np.vstack([dataForClassification, np.squeeze(featuresData[:, :, target, :]).T])
            
        # print("trainData shape (2), ",dataForClassification.shape)
    
        dataForClassification = np.reshape(dataForClassification, (dataForClassification.shape[0], dataForClassification.shape[1], 
                                             dataForClassification.shape[2], 1))
        
        # print("Final trainData shape (3), ",dataForClassification.shape)
        
        return dataForClassification
    
    def getMagnitudFeatureVector(self, signal):
        
        magnitudFeatures = self.computeMSF(signal)
        
        return self.getDataForClassification(magnitudFeatures)
    
    def getComplexFeatureVector(self, signal):
        
        complexFeatures = self.computeCSF(signal)
        
        return self.getDataForClassification(complexFeatures)
        
        
def main():

    import fileAdmin as fa
    
    # actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    actualFolder = "E:/reposBCICompetition/BCIC-Personal/talleres/taller4/scripts"
    path = os.path.join(actualFolder,"dataset")
    
    subjects = [8]
    subjectsNames = ["s8"]
        
    fm = 256.0 #IMPORTANTE: La frecuencia de muestreo DEBE SER LA MISMA que se uso cuando se entrenó la CNN
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25,
                              9.75, 11.75, 13.75,
                              10.25, 12.25, 14.25,
                              10.75, 12.75, 14.75])
    
    """
    **********************************************************************
    Loading and plotting the EEG
    IMPORTANT: In real time BCI, the "loading data" will be replaced
    for real data coming from the OpenBCI board
    **********************************************************************
    """
    
    rawEEG = fa.loadData(path = path, filenames = subjectsNames)[f"s{subjects[0]}"]["eeg"]
    
    #selec the last 3 trials
    rawEEG = rawEEG[:, :, :, 12:]
    
    stimulus = 12 #slected stimulus for classification
    trial = 1 #selected trial
     
    #get the selected trial and stimulus from rawEEG
    data = rawEEG[stimulus-1,:,:,trial-1].reshape(1, rawEEG.shape[1],rawEEG.shape[2],1)
    path = os.path.join(actualFolder,"models")
    
    samples = rawEEG.shape[2]
    # resolution = fm/samples #IMPORTANTE: La resolución DEBE SER LA MISMA que se uso cuando se entrenó la CNN
    resolution = np.round(fm/samples,4) #IMPORTANTE: La resolución DEBE SER LA MISMA que se uso cuando se entrenó la CNN
    
    rawEEG = rawEEG[:,:, muestraDescarte: ,:]
    rawEEG = rawEEG[:,:, :tiempoTotal ,:]
    
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
    
    """
    **********************************************************************
    First step: Create CNNClassify object in order to load a trained model
    and classify new data
    **********************************************************************
    """
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    # actualFolder = "E:/reposBCICompetition/BCIC-Personal/talleres/taller4/scripts"
    path = os.path.join(actualFolder,"dataset")
    
    # create an CNNClassify object in order to work with magnitud features
    magnitudCNNClassifier = CNNClassify(modelFile = "CNN_UsingMagnitudFeatures_Subject8",
                                weightFile = "bestWeightss_CNN_UsingMagnitudFeatures_Subject8",
                                frecStimulus = frecStimulus.tolist(),
                                PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                FFT_PARAMS = FFT_PARAMS,
                                classiName = f"CNN_Classifier")
    
    # create an CNNClassify object in order to work with magnitud and complex features
    complexCNNClassifier = CNNClassify(modelFile = "CNN_UsingComplexFeatures_Subject8",
                                weightFile = "bestWeightss_CNN_UsingComplexFeatures_Subject8",
                                frecStimulus = frecStimulus.tolist(),
                                PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                FFT_PARAMS = FFT_PARAMS,
                                classiName = f"CNN_Classifier")
    
    """
    **********************************************************************
    Second step: Get the feature vector
    **********************************************************************
    """
    
    featureVector = magnitudCNNClassifier.getMagnitudFeatureVector(data)
    
    # Get a classification. The classifyEEGSignal() method give us a stimulus
    # complexClassification = complexCNNClassifier.classifyEEGSignal(complexDataForClassification)
    magnitudClassification = magnitudCNNClassifier.classifyEEGSignal(featureVector)
    
    
    print("The stimulus classified using magnitud features is: ", magnitudClassification)
    # print("The stimulus classified using complex features is: ", complexClassification)
    
    plotOneSpectrum(magnitudCNNClassifier.MSF, resolution, 12, subjects[0], 5, [magnitudClassification],
                  startFrecGraph = FFT_PARAMS['start_frequency'],
                  save = False,
                  title = f"Stimulus classified using magnitud features: {magnitudClassification}", folder = "figs")
    
    trials = 3
    predicciones = np.zeros((len(frecStimulus),trials))
    
    for i, stimulus in enumerate(np.arange(12)):
        for j, trial in enumerate(np.arange(3)):
            data = rawEEG[stimulus,:,:,trial].reshape(1, rawEEG.shape[1],rawEEG.shape[2],1)
            featureVector = magnitudCNNClassifier.getMagnitudFeatureVector(data)
            classification = magnitudCNNClassifier.classifyEEGSignal(featureVector)
            if classification == frecStimulus[stimulus]:
                predicciones[i,j] = 1
    
    predMag = pd.DataFrame(predicciones, index = frecStimulus,
                           columns = [f"trial {trial+1}" for trial in np.arange(trials)])
    
    print("Predicciones usando como features la magnitud de la FFT")
    print( predMag)

    predicciones = np.zeros((len(frecStimulus),trials))    
    for i, stimulus in enumerate(np.arange(12)):
        for j, trial in enumerate(np.arange(3)):
            data = rawEEG[stimulus,:,:,trial].reshape(1, rawEEG.shape[1],rawEEG.shape[2],1)
            featureVector = magnitudCNNClassifier.getComplexFeatureVector(data)
            # features = complexCNNClassifier.computeCSF(data)
            # dataForClassification = complexCNNClassifier.getDataForClassification(features)
            classification = complexCNNClassifier.classifyEEGSignal(featureVector)
            if classification == frecStimulus[stimulus]:
                predicciones[i,j] = 1
        
    predCom = pd.DataFrame(predicciones, index = frecStimulus,
                           columns = [f"trial {trial+1}" for trial in np.arange(trials)])
    
    print("Predicciones usando features con parte real e imaginaria")
    print(predCom)
    

if __name__ == "__main__":
    main()


