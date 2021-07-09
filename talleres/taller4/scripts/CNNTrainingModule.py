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

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

from brainflow.data_filter import DataFilter

#own packages

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum

# warnings.filterwarnings('ignore')

class CNNTrainingModule():
    
    def __init__(self, rawEEG, subject = "1", PRE_PROCES_PARAMS = dict(), FFT_PARAMS = dict(), CNN_PARAMS = dict(),
                 modelName = ""):
        
        """
        Some important variables configuration and initialization in order to implement a CNN model.
        
        The model was proposed in 'Comparing user-dependent and user-independent
        training of CNN for SSVEP BCI' study.
        
        The rawEEG data is expected as
        [Number of targets, Number of channels, Number of sampling points, Number of trials]
        
        
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

        
        """
        
        self.rawEEG = rawEEG
        self.subject = subject
        
        if not modelName:
            self.modelName = f"CNNModelSubject{subject}"
            
        self.modelName = modelName
        
        self.eeg_channels = self.rawEEG.shape[0]
        self.total_trial_len = self.rawEEG.shape[2]
        self.num_trials = self.rawEEG.shape[3]
        
        self.modelSummary = None
        self.model = None
        self.bestWeights = None
            
        self.all_acc = np.zeros((10, 1)) #Accuracy values
        
        self.MSF = np.array([]) #Magnitud Spectrum Features
        self.CSF = np.array([]) #Momplex Spectrum Features
        
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
        
        
    def computeMSF(self):
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
        filteredEEG = filterEEG(self.rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["sampling_rate"])
        
        #eeg data segmentation
        eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                     self.PRE_PROCES_PARAMS["shiftLen"],
                                     self.PRE_PROCES_PARAMS["sampling_rate"])
        
        print("eegSegmented, ", eegSegmented.shape)
        
        self.CSF = computeComplexSpectrum(eegSegmented, self.FFT_PARAMS)
        
        return self.CSF
    
    def getDataForTraining(self, features):
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
        
        trainingData = featuresData[:, :, 0, :].T
        # print("Transpose trainData shape(1), ", trainingData.shape)
        
        #Reshaping the data into dim [classes*trials x channels x features]
        for target in range(1, featuresData.shape[2]):
            trainingData = np.vstack([trainingData, np.squeeze(featuresData[:, :, target, :]).T])
            
        # print("trainData shape (2), ",trainingData.shape)
    
        trainingData = np.reshape(trainingData, (trainingData.shape[0], trainingData.shape[1], 
                                             trainingData.shape[2], 1))
        
        # print("Final trainData shape (3), ",trainingData.shape)
        
        epochsPerClass = featuresData.shape[3]
        featuresData = []
        
        classLabels = np.arange(self.CNN_PARAMS['num_classes'])
        labels = (npm.repmat(classLabels, epochsPerClass, 1).T).ravel()
        
        labels = to_categorical(labels)     
        
        # print(labels[:,1])
        # print("Labels shape: ", labels.shape)
        
        return trainingData, labels
    
    def CNN_model(self,inputShape):
        '''
        
        Make the CNN model
    
        Args:
            inputShape (numpy.ndarray): shape of input training data with form
            [Number of Channels x Number of feates x 1]
    
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
    
    def trainCNN(self, trainingData, labels, nFolds = 10, saveBestWeights = True):
        """
        Perform a CNN training using a cross validation method.
        
        Arguments:
            - trainingData: Data to use in roder to train the CNN with shape
            e.g. [num_training_examples, num_channels, n_fc] or [num_training_examples, num_channels, 2*n_fc].
            - labels: The labels for data training.
            - nFolds: Number of folds for he cross validation.
            - saveBestWeights = True: If we want to save the best weights for the CNN training
            
        Return:
            - Accuracy for the CNN model using the trainingData
        """
        
        kf = KFold(n_splits = nFolds, shuffle=True)
        kf.get_n_splits(trainingData)
        accu = np.zeros((nFolds, 1))
        fold = -1
        
        score = 0.0

        if not self.model: #Check if themodel is empty
            print("Empty model. You should invoke createModel() method first")
            
        else:
    
            for trainIndex, testIndex in kf.split(trainingData):
                
                xValuesTrain, xValuesTest = trainingData[trainIndex], trainingData[testIndex]
                yValuesTrain, yValuesTest = labels[trainIndex], labels[testIndex]
                
                fold = fold + 1
                print(f"Subject: {self.subject} - Fold: {fold+1} Training...")
                
                sgd = optimizers.SGD(lr = self.CNN_PARAMS['learning_rate'],
                                      decay = self.CNN_PARAMS['lr_decay'],
                                      momentum = self.CNN_PARAMS['momentum'], nesterov=False)
                
                self.model.compile(loss = categorical_crossentropy, optimizer = sgd, metrics = ["accuracy"])
                
                history = self.model.fit(xValuesTrain, yValuesTrain, batch_size = self.CNN_PARAMS['batch_size'],
                                    epochs = self.CNN_PARAMS['epochs'], verbose=0)
        
                actualSscore = self.model.evaluate(xValuesTest, yValuesTest, verbose=0) 
                
                if saveBestWeights:
                    
                    try:
                        actualFolder = os.getcwd()
                        os.makedirs("models")    
                        print("Directory 'models' created ")
                    except FileExistsError:
                        print("")
                        
                    if actualSscore[1] > score:
                        score = actualSscore[1]
                        self.model.save_weights(f'models/bestWeightss_{self.modelName}.h5')
                
                accu[fold, :] = actualSscore[1]*100
                
                print("%s: %.2f%%" % (self.model.metrics_names[1], actualSscore[1]*100))
                
            print(f"Mean accuracy for overall folds for subject {self.subject}: {np.mean(accu)}")
            
            return accu
    
    def saveCNNModel(self):
        """
        Save the model created.
        
        Argument: None. This method use variables from the own class
        """
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#save
        
        if not self.model: #Check if themodel is empty
            print("Empty model")
            
        else:
            try:
                actualFolder = os.getcwd()
                os.makedirs("models")    
                print("Directory 'models' created ")
            except FileExistsError:
                print("")
                
            self.model.save(f"models/{self.modelName}.h5")
            modelInJson = self.model.to_json()
            with open(f"models/{self.modelName}.json", "w") as jsonFile:
                jsonFile.write(modelInJson)
    
def main():
        
    import fileAdmin as fa
            
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"dataset")
    # dataSet = sciio.loadmat(f"{path}/s{subject}.mat")
    
    # path = "E:/reposBCICompetition/BCIC-Personal/taller4/scripts/dataset" #directorio donde estan los datos
    
    subjects = [8]
    
    fm = 256.0
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
    
    rawEEG = fa.loadData(path = path, subjects = subjects)[f"s{subjects[0]}"]["eeg"]
    
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
    
    
    #Make a CNNTrainingModule object in order to use the Magnitude Features of the data
    magnitudCNN = CNNTrainingModule(rawEEG = rawEEG, subject = subjects[0],
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    CNN_PARAMS = CNN_PARAMS,
                                    modelName = f"CNN_UsingMagnitudFeatures_Subject{subjects[0]}")
    
    #Computing and getting the magnitude Spectrum Features
    magnitudFeatures = magnitudCNN.computeMSF()
    
    plotSpectrum(magnitudFeatures, resolution, 12, subjects[0], 7, frecStimulus,
                  startFrecGraph = FFT_PARAMS['start_frequency'],
                  save = False, title = "", folder = "figs")
    
    
    # Get the training and testing data for CNN using Magnitud Spectrum Features
    trainingData_MSF, labels_MSF = magnitudCNN.getDataForTraining(magnitudFeatures)
    
    #inputshape [Number of Channels x Number of Features x 1]
    inputshape = np.array([trainingData_MSF.shape[1], trainingData_MSF.shape[2], trainingData_MSF.shape[3]])
    #Create the CNN model
    magnitudCNN.createModel(inputshape)
    #Training the CNN model suing the training data
    accu_CNN_using_MSF = magnitudCNN.trainCNN(trainingData_MSF, labels_MSF, nFolds = 10)
    
    #save the model
    magnitudCNN.saveCNNModel()
    
    """Make a CNNTrainingModule object in order to use the Magnitude and Complex Features of the data"""
    complexCNN = CNNTrainingModule(rawEEG = rawEEG, subject = subjects[0],
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    CNN_PARAMS = CNN_PARAMS,
                                    modelName = f"CNN_UsingComplexFeatures_Subject{subjects[0]}")
    
    complexFeatures = complexCNN.computeCSF()
    
    # Training and testing CNN suing Complex Spectrum Features
    trainingData_CSF, labels_CSF = complexCNN.getDataForTraining(complexFeatures)
    
    #inputshape [Number of Channels x Number of Features x 1]
    inputshape = np.array([trainingData_CSF.shape[1], trainingData_CSF.shape[2], trainingData_CSF.shape[3]])
    
    # Create the CNN model
    complexCNN.createModel(inputshape)
    
    accu_CNN_using_CSF = complexCNN.trainCNN(trainingData_CSF, labels_CSF, nFolds = 10)
    
    complexCNN.saveCNNModel()
        
    
if __name__ == "__main__":
    main()



        
