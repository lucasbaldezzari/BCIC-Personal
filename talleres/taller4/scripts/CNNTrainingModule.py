# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:53:29 2021

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy

from brainflow.data_filter import DataFilter

#own packages

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum

# warnings.filterwarnings('ignore')

class CNNTrainingModule():
    
    def __init__(self, rawEEG, subject = "1", PRE_PROCES_PARAMS = dict(), FFT_PARAMS = dict(), CNN_PARAMS = dict()):
        
        """
        The rawEEG data is expected as
        [Number of targets, Number of channels, Number of sampling points, Number of trials]
        """
        
        self.rawEEG = rawEEG
        self.subject = subject
        
        self.eeg_channels = self.rawEEG.shape[0]
        self.total_trial_len = self.rawEEG.shape[2]
        self.num_trials = self.rawEEG.shape[3]
        
        # self.sampling_rate = sampling_rate
        
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
             
        if not FFT_PARAMS:        
            self.FFT_PARAMS = {
                'resolution': 0.2930,
                'start_frequency': 0.0,
                'end_frequency': 38.0,
                'sampling_rate': 256.
                }
        else:
            self.FFT_PARAMS = FFT_PARAMS
        
        
        self.window_len = 1
        self.shift_len = 1
            
        self.all_acc = np.zeros((10, 1))
        
        self.MSF = np.array([]) #Magnitud Spectrum Features
        self.CSF = np.array([]) #Momplex Spectrum Features
        
        self.mcnn_training_data = dict()
        self.ccnn_training_data = dict()
        
        self.mcnn_results = dict()
        self.ccnn_results = dict()
        
        self.segmentedEEG = list()
        
    def computeMSF(self):
        
        #eeg data filtering
        filteredEEG = filterEEG(self.rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["sampling_rate"])
        
        #eeg data segmentation
        eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                     self.PRE_PROCES_PARAMS["shiftLen"],
                                     self.PRE_PROCES_PARAMS["sampling_rate"])
        
        self.MSF = computeMagnitudSpectrum(eegSegmented, self.FFT_PARAMS)
        
        return self.MSF

    def getMSF(self):
        
        # return MSF
        return self.MSF
    
    def computeCSF(self):
        
        #eeg data filtering
        filteredEEG = filterEEG(self.rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["sampling_rate"])
        
        #eeg data segmentation
        eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                     self.PRE_PROCES_PARAMS["shiftLen"],
                                     self.PRE_PROCES_PARAMS["sampling_rate"])
        
        self.CSF = computeComplexSpectrum(eegSegmented, self.FFT_PARAMS)
        
        return self.CSF
    
    def getDataForTraining(self, MSF):
        
        # computeMagnitudSpectrum
        print("Generating training data")
        # print("Original MSF shape: ", MSF.shape)
        featuresData = np.reshape(MSF, (MSF.shape[0], MSF.shape[1],MSF.shape[2],
                                        MSF.shape[3]*MSF.shape[4]))
        # print("featuresData shape: ", featuresData.shape)
        
        trainingData = featuresData[:, :, 0, :].T
        # print("trainData shape (1), ",trainingData.shape)
        
        #Reshaping the data into dim [classes*trials*segments X channels X features]
        for target in range(1, featuresData.shape[2]):
            trainingData = np.vstack([trainingData, np.squeeze(featuresData[:, :, target, :]).T])
            
        # print("trainData shape (2), ",trainingData.shape)
    
        trainingData = np.reshape(trainingData, (trainingData.shape[0], trainingData.shape[1], 
                                             trainingData.shape[2], 1))
        
        # print("trainData shape (3), ",trainingData.shape)
        
        epochsPerClass = featuresData.shape[3]
        featuresData = []
        
        classLabels = np.arange(self.CNN_PARAMS['num_classes'])
        labels = (npm.repmat(classLabels, epochsPerClass, 1).T).ravel()
        
        labels = to_categorical(labels)        
        
        return trainingData, labels
    
    def CNN_model(self,inputShape):
        '''
        Returns the Concolutional Neural Network model for SSVEP classification.
    
        Args:
            inputShape (numpy.ndarray): shape of input training data 
            e.g. [num_training_examples, num_channels, n_fc] or [num_training_examples, num_channels, 2*n_fc].
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
    
    def trainCNN(self, trainingData, labels, nFolds = 10):
        """
        Perform a CNN training using a cross validation method.
        
        Arguments:
            - trainingData: Data to use in roder to train the CNN
            - labels: The labels for my data.
            - nFolds: Number of folds for he cross validation.
            
        Return:
            - Accuracy for the CNN model using the trainingData
        """
        
        kf = KFold(n_splits = nFolds, shuffle=True)
        kf.get_n_splits(trainingData)
        accu = np.zeros((nFolds, 1))
        fold = -1
    
        for trainIndex, testIndex in kf.split(trainingData):
            
            xValuesTrain, xValuesTest = trainingData[trainIndex], trainingData[testIndex]
            yValuesTrain, yValuesTest = labels[trainIndex], labels[testIndex]
            inputShape = np.array([xValuesTrain.shape[1], xValuesTrain.shape[2], xValuesTrain.shape[3]])
            
            fold = fold + 1
            print(f"Subject: {self.subject} - Fold: {fold+1} Training...")
            
            model = self.CNN_model(inputShape)
            
            sgd = optimizers.SGD(lr = self.CNN_PARAMS['learning_rate'],
                                  decay = self.CNN_PARAMS['lr_decay'],
                                  momentum = self.CNN_PARAMS['momentum'], nesterov=False)
            
            model.compile(loss = categorical_crossentropy, optimizer = sgd, metrics = ["accuracy"])
            
            history = model.fit(xValuesTrain, yValuesTrain, batch_size = self.CNN_PARAMS['batch_size'],
                                epochs = self.CNN_PARAMS['epochs'], verbose=0)
    
            score = model.evaluate(xValuesTest, yValuesTest, verbose=0) 
            
            accu[fold, :] = score[1]*100
            
            print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
            
        print(f"Mean accuracy for overall folds for subject {self.subject}: {np.mean(accu)}")
        
        return accu
        
    
    def saveMSF(self, MSFs, fileName = "magnitudeSpectrumFeatures"):
        """Save the Magnitude Spectrum Features in a txt file"""
        
        print(f"Saving file like {fileName}.txt")
        
        # points = MSFs.shape[0]
        points = MSFs.size
        shape = MSFs.shape
        
        # for index in range(len(shape)-1):
        #     points = points*shape[index+1]

        #Save the MSF in a file. NOTE: there are many numbers, so could take a while save the entire data in the file
        np.savetxt(f"{fileName}.txt", MSFs.reshape(1,points))
        np.savetxt(f"{fileName}_OrginalShape.txt", shape)
        
    def loadMSF(self, fileName = "magnitudeSpectrumFeatures", dataShape = (110,8,12,1114,5)):
        """Load the Magnitude Spectrum Features from a txt file
        
        Return:
        Magnitude Spectrum Features with shape
        [n_fc, num_channels, num_classes, num_trials, number_of_segments]        
        """
        
        print(f"Loading file {fileName}.txt")
        originalShape = tuple(int(i) for i in np.loadtxt(f"{fileName}_OrginalShape.txt"))
        
        return np.loadtxt(f"{fileName}.txt").reshape(originalShape)

    
def main():
        
    import fileAdmin as fa
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"dataset")
    # dataSet = sciio.loadmat(f"{path}/s{subject}.mat")
    
    # path = "E:/reposBCICompetition/BCIC-Personal/taller4/scripts/dataset" #directorio donde estan los datos
    
    subjects = [8]
    
    fm = 256
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
    
    PRE_PROCES_PARAMS = {
                    'lfrec': 3.,
                    'hfrec': 80.,
                    'order': 4,
                    'sampling_rate': 256.,
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
                    'resolution': 0.2930,
                    'start_frequency': 0.0,
                    'end_frequency': 38.0,
                    'sampling_rate': 256.
                    }
    
    rawEEG = fa.loadData(path = path, subjects = subjects)[f"s{subjects[0]}"]["eeg"]
    rawEEG = rawEEG[:,:, muestraDescarte: ,:]
    rawEEG = rawEEG[:,:, :tiempoTotal ,:]
    
    #Make a CNNTrainingModule object
    trainingCNN = CNNTrainingModule(rawEEG = rawEEG, subject = subjects[0],
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    CNN_PARAMS = CNN_PARAMS)
    
    #Computing and getting the magnitude Spectrum Features
    magnitudFeatures = trainingCNN.computeMSF()
    
    plotSpectrum(trainingCNN.MSF, 0.2930, 12, subjects[0], 7, frecStimulus,
                      save = False, title = "", folder = "figs")
    
    #Training and testing CNN suing Magnitud Spectrum Features
    trainingData_MSF, labels_MSF = trainingCNN.getDataForTraining(magnitudFeatures)
    # accu_CNN_using_MSF = trainingCNN.trainCNN(trainingData_MSF, labels_MSF)
    
    complexFeatures = trainingCNN.computeCSF()
    #Training and testing CNN suing Magnitud Spectrum Features
    trainingData_CSF, labels_CSF = trainingCNN.getDataForTraining(complexFeatures)
    accu_CNN_using_CSF = trainingCNN.trainCNN(trainingData_CSF, labels_CSF)
        
    
if __name__ == "__main__":
    main()

    
        
        
