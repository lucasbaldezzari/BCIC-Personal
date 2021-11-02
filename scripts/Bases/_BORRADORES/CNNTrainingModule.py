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
import fileAdmin as fa

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

#own packages

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum

# warnings.filterwarnings('ignore')

class CNNTrainingModule():
    
    def __init__(self, rawEEG, subject = "1", PRE_PROCES_PARAMS = dict(), FFT_PARAMS = dict(),
                 CNN_PARAMS = dict(),
                 modelName = ""):
        
        """
        Some important variables configuration and initialization in order to implement a CNN model for training.
        
        The model was proposed in 'Comparing user-dependent and user-independent
        training of CNN for SSVEP BCI' study.
        
        Args:
            - rawEEG: Raw EEG. The rawEEG data is expected as
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
        
        Return: The Magnitud Spectrum Feature (MSF).
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
    
    def computeCSF(self):
        """
        Compute the FFT over segmented EEG data.
        
        Argument: None. This method use variables from the own class
        
        Return: The Complex Spectrum Feature (CSF) and the MSF in the same matrix
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
        
        self.CSF = computeComplexSpectrum(eegSegmented, self.FFT_PARAMS)
        
        return self.CSF
    
    def getDataForTraining(self, features):
        """
        Prepare the features set in order to fit and train the CNN model.
        
        Arguments:
            - features: Magnitud Spectrum Features or Complex Spectrum Features
        """
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
            [Number of Channels x Number of features x 1]
    
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
        
        listaAccu = []
        

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
                
                # print(history.history.keys())
                
                if saveBestWeights:
                    
                    try:
                        actualFolder = os.getcwd()
                        os.makedirs("models/cnn")    
                        print("Directory 'models' created ")
                    except FileExistsError:
                        print("")
                        
                    if actualSscore[1] > score:
                        score = actualSscore[1]
                        self.model.save_weights(f'models//cnn/bestWeightss_{self.modelName}.h5')
                
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
                os.makedirs("models/cnn")    
                print("Directory 'models/cnn' created ")
            except FileExistsError:
                print("")
                
            self.model.save(f"models/cnn/{self.modelName}.h5")
            modelInJson = self.model.to_json()
            with open(f"models/{self.modelName}.json", "w") as jsonFile:
                jsonFile.write(modelInJson)
    
def main():
        
    """Empecemos"""

    """
    **********************************************************************
    First step: Loading the EEG
    **********************************************************************
    """

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
    Second step: Create the CNN
    **********************************************************************
    """
    sujeto = "lucasB"
    #Make a CNNTrainingModule object in order to use the data's Magnitude Features
    magnitudCNN = CNNTrainingModule(rawEEG = trainSet, subject = 1,
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    CNN_PARAMS = CNN_PARAMS,
                                    modelName = f"CNN_UsingMagnitudFeatures_Subject{sujeto}")
    
    """
    **********************************************************************
    Third step: Compute and get the Magnitud Spectrum Features
    **********************************************************************
    """
    #Computing and getting the magnitude Spectrum Features
    magnitudFeatures = magnitudCNN.computeMSF()
    
    ### Descomentar las líneas de abajo para graficar el espectro
    
    plotSpectrum(magnitudFeatures, resolution, blancos = 4, sujeto = sujeto, canal = 1, frecStimulus = frecStimulus,
                  startFrecGraph = FFT_PARAMS['start_frequency'],
                  save = False, title = "", folder = "figs",
                  rows = 2, columns = 2)
    
    
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
    
    accu_CNN_using_MSF = magnitudCNN.trainCNN(trainingData_MSF, labels_MSF, nFolds = 6)
    print(f"Maxima accu {accu_CNN_using_MSF.max()}")
    
    #saving the model
    magnitudCNN.saveCNNModel()
    
    """
    **************************************
                  END           
    **************************************
    """
    
    
    """
    **************************************
    Make the same as before, but now using Magnitud and Complex features from the EEG
    **************************************
    """
    """Make a CNNTrainingModule object in order to use the data's Magnitude and data's Complex Features"""
    complexCNN = CNNTrainingModule(rawEEG = trainSet, subject = 1,
                                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                    FFT_PARAMS = FFT_PARAMS,
                                    CNN_PARAMS = CNN_PARAMS,
                                    modelName = f"CNN_UsingComplexFeatures_Subject{sujeto}")
    
    complexFeatures = complexCNN.computeCSF()
    
    # Training and testing CNN suing Complex Spectrum Features
    trainingData_CSF, labels_CSF = complexCNN.getDataForTraining(complexFeatures)
    
    #inputshape [Number of Channels x Number of Features x 1]
    inputshape = np.array([trainingData_CSF.shape[1], trainingData_CSF.shape[2], trainingData_CSF.shape[3]])
    
    # Create the CNN model
    complexCNN.createModel(inputshape)
    
    accu_CNN_using_CSF = complexCNN.trainCNN(trainingData_CSF, labels_CSF, nFolds = 6)
    print(f"Maxima accu {accu_CNN_using_CSF.max()}")
    
    complexCNN.saveCNNModel()
        
# if __name__ == "__main__":
#     main()


