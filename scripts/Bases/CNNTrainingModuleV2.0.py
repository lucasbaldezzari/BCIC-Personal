# -*- coding: utf-8 -*-

"""
Created on Thu Jun 24 16:53:29 2021

@author: Lucas BALDEZZARI

        VERSIÓN: SCT-01-RevB
"""

import os

import warnings
import numpy as np
import numpy.matlib as npm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

import pickle

#own packages

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum

# warnings.filterwarnings('ignore')

class CNNTrainingModule():
    
    def __init__(self, rawDATA, PRE_PROCES_PARAMS, FFT_PARAMS, CNN_PARAMS, frecStimulus, nchannels,nsamples,ntrials,modelName = ""):
        
        """
        Some important variables configuration and initialization in order to implement a CNN model for training.
        
        The model was proposed in 'Comparing user-dependent and user-independent
        training of CNN for SSVEP BCI' study.
        
        Args:
            - rawDATA: Raw EEG. The rawDATA data is expected as
            [Number of targets, Number of channels, Number of sampling points, Number of trials]
            - subject: Subject number
            - PRE_PROCES_PARAMS: The params used in order to pre process the raw EEG.
            - FFT_PARAMS: The params used in order to compute the FFT
            - CNN_PARAMS: The params used for the CNN model.
                CNN_PARAMS (dict): dictionary of parameters used for feature extraction.        
                CNN_PARAMS['batch_size'] (int): training mini batch size.
                CNN_PARAMS['epochs'] (int): total number of training epochs/iterations.
                CNN_PARAMS['droprate'] (float): dropout ratio.
                CNN_PARAMS['learning_rate'] (float): model learning rate.
                CNN_PARAMS['lr_decay'] (float): learning rate decay ratio.
                CNN_PARAMS['l2_lambda'] (float): l2 regularization parameter.
                CNN_PARAMS['momentum'] (float): momentum term for stochastic gradient descent optimization.
                CNN_PARAMS['kernel_f'] (int): 1D kernel to operate on conv_1 layer for the SSVEP CNN. 
                CNN_PARAMS['n_ch'] (int): number of eeg channels
                CNN_PARAMS['num_classes'] (int): number of SSVEP targets/classes
            - modelName: The model name used to identify the object and the model        
        """
        self.rawDATA = rawDATA

        if not modelName:
            self.modelName = f"CNNModelSubject"

        else:
            self.modelName = modelName

        #Setting variables for EEG processing.
        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
            
        #Setting variables for CNN training
        self.CNN_PARAMS = CNN_PARAMS
             
        #Setting variables for the Magnitude and Complex features extracted from FFT
        self.FFT_PARAMS = FFT_PARAMS

        self.frecStimulus = frecStimulus
        self.nclases = len(frecStimulus)
        self.nchannels = nchannels #ex nchannels
        self.nsamples = nsamples #ex nsamples
        self.ntrials = ntrials #ex ntrials

        self.dataBanked = None #datos de EEG filtrados con el banco
        self.model = None #donde almacenaremos el modelo CNN
        self.trainingData = None
        self.labels = None

        self.signalPSD = None #Power Spectrum Density de mi señal
        self.signalSampleFrec = None
       
        self.modelSummary = None
        self.model = None
        self.bestWeights = None
            
        self.all_acc = np.zeros((10, 1)) #Accuracy values
        
        self.MSF = np.array([]) #Magnitud Spectrum Features
        self.CSF = np.array([]) #Momplex Spectrum Features

    def CNN_model(self,inputShape):
        '''
        
        Make the CNN model
    
        Args:
            inputShape (numpy.ndarray): shape of input training data with form
            [Number of Channels x Number of features]
    
        Returns:
            (keras.Sequential): CNN model.
        '''
        
        model = Sequential()
        
        model.add(Conv2D(2*self.CNN_PARAMS['n_ch'], kernel_size=(self.CNN_PARAMS['n_ch'], 1),
                         input_shape=(inputShape[0], inputShape[1], inputShape[2]),
                         padding="valid", kernel_regularizer=regularizers.l2(self.CNN_PARAMS['l2_lambda']),
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        
        model.add(BatchNormalization())
        
        model.add(Activation('relu'))
        
        model.add(Dropout(self.CNN_PARAMS['droprate']))  
        
        model.add(Conv2D(2*self.CNN_PARAMS['n_ch'], kernel_size=(1, self.CNN_PARAMS['kernel_f']), 
                         kernel_regularizer = regularizers.l2(self.CNN_PARAMS['l2_lambda']), padding="valid", 
                         kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        
        model.add(BatchNormalization())
        
        model.add(Activation('relu'))
        
        model.add(Dropout(self.CNN_PARAMS['droprate']))  
        
        model.add(Flatten())
        
        model.add(Dense(self.CNN_PARAMS['num_classes'], activation='softmax', 
                        kernel_regularizer=regularizers.l2(self.CNN_PARAMS['l2_lambda']), 
                        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        
        return model
    
    def createModel(self, inputShape):
        """
        Create the CNN Model using inputShape.
        The CNN expected inputshape like [Channels x Number of Features x 1]
        """
        
        #Create the CNN model using the inputShape
        self.model = self.CNN_model(inputShape)
        
        self.modelSummary = self.model.summary()

    def applyFilterBank(self, eeg, bw = 2.0, order = 4):
        """Aplicamos banco de filtro a nuestros datos.
        Se recomienda aplicar un notch en los 50Hz y un pasabanda en las frecuencias deseadas antes
        de applyFilterBank()
        
        Args:
            - eeg: datos a aplicar el filtro. Forma [clase, samples, trials]
            - frecStimulus: lista con la frecuencia central de cada estímulo/clase
            - bw: ancho de banda desde la frecuencia central de cada estímulo/clase. Default = 2.0
            - order: orden del filtro. Default = 4"""

        nyquist = 0.5 * self.FFT_PARAMS["sampling_rate"]
        signalFilteredbyBank = np.zeros((self.nclases,self.nsamples,self.ntrials))
        for clase, frecuencia in enumerate(self.frecStimulus):   
            low = (frecuencia-bw/2)/nyquist
            high = (frecuencia+bw/2)/nyquist
            b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
            central = filtfilt(b, a, eeg[clase], axis = 0)
            b, a = butter(order, [low*2, high*2], btype='band') #obtengo los parámetros del filtro
            firstHarmonic = filtfilt(b, a, eeg[clase], axis = 0)
            # signalFilteredbyBank[clase] = filtfilt(b, a, eeg[clase], axis = 0) #filtramos
            signalFilteredbyBank[clase] = central + firstHarmonic

        self.dataBanked = signalFilteredbyBank

        return self.dataBanked

    def computWelchPSD(self, signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):

        self.signalSampleFrec, self.signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = axis)

        return self.signalSampleFrec, self.signalPSD


    def featuresExtraction(self, ventana, anchoVentana = 5, bw = 2.0, order = 4, axis = 1):

        filteredEEG = filterEEG(self.rawDATA, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["bandStop"],
                                self.PRE_PROCES_PARAMS["sampling_rate"],
                                axis = axis)

        dataBanked = self.applyFilterBank(filteredEEG, bw=bw, order = 4)

        anchoVentana = int(self.PRE_PROCES_PARAMS["sampling_rate"]*anchoVentana) #fm * segundos
        ventana = ventana(anchoVentana)

        self.signalSampleFrec, self.signalPSD = self.computWelchPSD(dataBanked,
                                                fm = self.PRE_PROCES_PARAMS["sampling_rate"],
                                                ventana = ventana, anchoVentana = anchoVentana,
                                                average = "median", axis = axis)

        return self.signalSampleFrec, self.signalPSD

def main():
        
    """Empecemos"""

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG\WM\ses1")

    frecStimulus = np.array([6, 7, 8, 9])

    trials = 15
    fm = 200.
    window = 5 #sec
    samplePoints = int(fm*window)
    channels = 4

    filesRun1 = ["S3_R1_S2_E6","S3-R1-S1-E7", "S3-R1-S1-E8","S3-R1-S1-E9"]
    run1 = fa.loadData(path = path, filenames = filesRun1)
    filesRun2 = ["S3_R2_S2_E6","S3-R2-S1-E7", "S3-R2-S1-E8","S3-R2-S1-E9"]
    run2 = fa.loadData(path = path, filenames = filesRun2)

    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'lfrec': 4.,
                    'hfrec': 38.,
                    'order': 8,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': window,
                    'shiftLen':window
                    }

    resolution = np.round(fm/samplePoints, 4)

    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 4.0,
                    'end_frequency': 38.0,
                    'sampling_rate': fm
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
                    'n_ch': 2,
                    'num_classes': 4}

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData #la forma de joinedData es [estímulos, canales, muestras, trials]

    run1JoinedData = joinData(run1, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    run2JoinedData = joinData(run2, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)

    trainSet = np.concatenate((run1JoinedData[:,:,:,:12], run2JoinedData[:,:,:,:12]), axis = 3)
    trainSet = trainSet[:,:2,:,:] #nos quedamos con los primeros dos canales

    """
    **********************************************************************
    Second step: Create the CNN model
    **********************************************************************
    """

    nsamples = trainSet.shape[1]
    ntrials = trainSet.shape[2]

    #Make a CNNTrainingModule object in order to use the data's Magnitude Features
    cnn = CNNTrainingModule(trainSet, PRE_PROCES_PARAMS = PRE_PROCES_PARAMS, FFT_PARAMS = FFT_PARAMS, CNN_PARAMS = CNN_PARAMS,
                        frecStimulus = frecStimulus, nchannels = 1,nsamples = nsamples, ntrials = ntrials, modelName = "cnntesting")

    # >>> trainingData_MSF, labels_MSF = magnitudCNN.getDataForTraining(magnitudFeatures)
    # >>> trainingData_MSF.shape
    # (96, 2, 171, 1)
    # >>> labels_MSF.shape
    # (96, 4)
