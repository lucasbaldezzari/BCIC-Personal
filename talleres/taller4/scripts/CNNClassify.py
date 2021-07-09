# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 19:08:42 2021

@author: Lucas BALDEZZARI
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
    
    def __init__(self, modelFile = "CNNModel", weightFile = "weightFile",
                 PRE_PROCES_PARAMS = dict(), FFT_PARAMS = dict(), CNN_PARAMS = dict(),
                 classiName = "",
                 frecStimulus = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]):
        
        
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
        if not PRE_PROCES_PARAMS:
            self.PRE_PROCES_PARAMS = {
                'lfrec': 3.,
                'hfrec': 80.,
                'order': 4,
                'sampling_rate': 256.,
                'window': 4,
                'shiftLen':4
                }
        else:
            self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
            
        #Setting variables for CNN training
        if not CNN_PARAMS:     
            self.CNN_PARAMS = {
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
        else:
             self.CNN_PARAMS = CNN_PARAMS
             
        #Setting variables for the Magnitude and Complex features extracted from FFT
        if not FFT_PARAMS:        
            self.FFT_PARAMS = {
                'resolution': 0.2930,
                'start_frequency': 0.0,
                'end_frequency': 38.0,
                'sampling_rate': 256.
                }
        else:
            self.FFT_PARAMS = FFT_PARAMS

        
    def classifyEEGSignal(self, dataForClassification):
        
        self.preds = self.loadedModel.predict(dataForClassification)
        
        # print(self.preds.shape)
        return self.frecStimulusList[np.argmax(self.preds[0])]
    
    def computeMSF(self, rawEEG):
            """
            Compute the FFT over segmented EEG data.
            
            Argument: None. This method use variables from the own class
            
            Return: The Magnitud Spectrum Feature (MSF). Only considers the magnitude at different
            frecuencies, without the phase information.
            The matrix returned has shape Nch × Nfc
            
            [ Re{FFT(xo1)}
              Re{FFT(xo2)}
                .
                .
                .
              Re{FFT(xon)} ]
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
            
            Argument: None. This method use variables from the own class
            
            Return: The Complex Spectrum Feature (CSF) and the MSF in the same matrix with shape 
            Nch × Nfc.
            
            [ Re{FFT(xo1), Im{FFT(xo1)}
              Re{FFT(xo2), Im{FFT(xo2)}
                .
                .
                .
              Re{FFT(xon), Im{FFT(xon)} ]
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
            
            self.CSF = computeComplexSpectrum(eegSegmented, self.FFT_PARAMS)
            
            return self.CSF

    def getDataForClassification(self, features):
        """
        Prepare the features set in order to fit the CNN model and get a classification.
        
        Arguments:
            - features: Magnitud Spectrum Features or Complex Spectrum Features with shape
            
            [targets, channels, trials, segments, samples].
        
        """
        
        print("Generating training data")
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
        
def main():
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"dataset")
    # dataSet = sciio.loadmat(f"{path}/s{subject}.mat")
    
# if __name__ == "__main__":
#     main()

import fileAdmin as fa

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"dataset")

subjects = [8]
    
fm = 256.0
tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
muestraDescarte = 39
frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])

rawEEG = fa.loadData(path = path, subjects = subjects)[f"s{subjects[0]}"]["eeg"]

stimulus = 12
trial = 1


data = rawEEG[stimulus-1,:,:,trial-1].reshape(1, rawEEG.shape[1],rawEEG.shape[2],1)

path = os.path.join(actualFolder,"models")

samples = rawEEG.shape[2]
resolution = fm/samples

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

CNNClassifier = CNNClassify(modelFile = "CNN_UsingComplexFeatures_Subject8",
                           weightFile = "bestWeightss_CNN_UsingComplexFeatures_Subject8",
                           PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                           FFT_PARAMS = FFT_PARAMS,
                           CNN_PARAMS = CNN_PARAMS,
                           classiName = f"CNN_Classifier",
                           frecStimulus = frecStimulus.tolist())

magnitudCNNClassifier = CNNClassify(modelFile = "CNN_UsingMagnitudFeatures_Subject8",
                           weightFile = "bestWeightss_CNN_UsingMagnitudFeatures_Subject8",
                           PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                           FFT_PARAMS = FFT_PARAMS,
                           CNN_PARAMS = CNN_PARAMS,
                           classiName = f"CNN_Classifier",
                           frecStimulus = frecStimulus.tolist())

magnitudFeatures = magnitudCNNClassifier.computeMSF(data)
complexFeatures = CNNClassifier.computeCSF(data)

complexDataForClassification = CNNClassifier.getDataForClassification(complexFeatures)
magnitudDataForClassification = magnitudCNNClassifier.getDataForClassification(magnitudFeatures)

complexClassification = CNNClassifier.classifyEEGSignal(complexDataForClassification)
magnitudClassification = magnitudCNNClassifier.classifyEEGSignal(magnitudDataForClassification)

print(f"The stimulus classified using magnitud features is: ", magnitudClassification)
print(f"The stimulus classified using complex features is: ", complexClassification)

plotOneSpectrum(magnitudFeatures, resolution, 12, subjects[0], 5, [magnitudClassification],
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = f"Stimulus classified {magnitudClassification}", folder = "figs")
