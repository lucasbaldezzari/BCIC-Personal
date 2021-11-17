# -*- coding: utf-8 -*-

"""
Created on Thu Jun 24 16:53:29 2021

@author: Lucas BALDEZZARI

        VERSIÓN: SCT-01-RevB
"""

import os

import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import fileAdmin as fa
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

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

    def CNN_model(self, inputShape):
        '''
        
        Make the CNN model
    
        Args:
            inputShape (numpy.ndarray): shape of input training data with form
            [Trials totales x Cantidad de features por trial]
    
        Returns:
            (keras.Sequential): CNN model.
        '''
        
        model = Sequential()
        
        model.add(Conv2D(2*self.CNN_PARAMS['n_ch'], kernel_size=(self.CNN_PARAMS['n_ch'], 1),
                         input_shape=(inputShape[0], inputShape[1], inputShape[2]),
                         padding="valid", kernel_regularizer=regularizers.l2(self.CNN_PARAMS['l2_lambda']),
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))

        # model.add(Conv2D(2*self.CNN_PARAMS['n_ch'], kernel_size=(self.CNN_PARAMS['n_ch'], 1),
        #                  input_shape=(inputShape[0], inputShape[1], inputShape[2]),
        #                  padding="valid", kernel_regularizer=regularizers.l2(self.CNN_PARAMS['l2_lambda']),
        #                  kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        
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
            signalFilteredbyBank[clase] = filtfilt(b, a, eeg[clase], axis = 0) #filtramos

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

    #Transforming data for training
    def getDataForTraining(self, features):
        """Preparación del set de entrenamiento.

        Argumentos:
            - features: Parte Real del Espectro or Parte Real e Imaginaria del Espectro
            con forma [número de características x canales x clases x trials x número de segmentos]
            - clases: Lista con las clases para formar las labels

        Retorna:
            - trainingData: Set de datos de entrenamiento para alimentar el modelo SVM
            Con forma [trials*clases x number of features]
            - Labels: labels para entrenar el modelo a partir de las clases
        """

        numFeatures = features.shape[1]
        trainingData = features.swapaxes(2,1).reshape(self.nclases*self.ntrials, numFeatures)

        classLabels = np.arange(self.nclases)

        labels = (npm.repmat(classLabels, self.ntrials, 1).T).ravel()

        labels = to_categorical(labels)

        return trainingData.reshape(self.nclases*self.ntrials,self.nchannels,numFeatures,1), labels

    def trainCNN(self, trainingData, labels, nFolds = 10, saveBestWeights = True, path = "models"):
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

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)
        
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
                print(f"Model: {self.modelName} - Fold: {fold+1} Training...")
                
                sgd = optimizers.SGD(lr = self.CNN_PARAMS['learning_rate'],
                                      decay = self.CNN_PARAMS['lr_decay'],
                                      momentum = self.CNN_PARAMS['momentum'], nesterov=False)
                
                self.model.compile(loss = categorical_crossentropy, optimizer = sgd, metrics = ["accuracy"])
                
                history = self.model.fit(xValuesTrain, yValuesTrain, batch_size = self.CNN_PARAMS['batch_size'],
                                    epochs = self.CNN_PARAMS['epochs'], verbose=0)
        
                actualSscore = self.model.evaluate(xValuesTest, yValuesTest, verbose=0)
                
                # print(history.history.keys())
                
                if saveBestWeights:
                    if actualSscore[1] > score:
                        score = actualSscore[1]
                        self.model.save_weights(f'bestWeightss_{self.modelName}.h5')
                
                accu[fold, :] = actualSscore[1]*100
                
                print("%s: %.2f%%" % (self.model.metrics_names[1], actualSscore[1]*100))
                
            print(f"Mean accuracy for overall folds for model {self.modelName}: {np.mean(accu)}")
            
            os.chdir(actualFolder)

            return accu

    def saveCNNModel(self, path, filename = ""):
        """
        Save the model created.
        
        Argument: None. This method use variables from the own class
        """
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#save
        
        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)

        if not filename:
            modelName = self.modelName
            filename = f"{self.modelName}.h5"
            
                
        self.model.save(f"{self.modelName}.h5")
        modelInJson = self.model.to_json()
        with open(f"{self.modelName}.json", "w") as jsonFile:
            jsonFile.write(modelInJson)

        #Guardamos los parámetros usados para entrenar el SVM
        file = open(f"{self.modelName}_preproces.json", "w")
        json.dump(self.PRE_PROCES_PARAMS , file)
        file.close

        file = open(f"{self.modelName}_fft.json", "w")
        json.dump(self.FFT_PARAMS , file)
        file.close

        os.chdir(actualFolder)

    def saveTrainingSignalPSD(self, signalPSD, path, filename = ""):

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)
        
        if not filename:
            filename = self.modelName

        np.savetxt(f'{filename}_signalPSD.txt', signalPSD, delimiter=',')
        np.savetxt(f'{filename}_signalSampleFrec.txt', self.signalSampleFrec, delimiter=',')

        os.chdir(actualFolder)

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

    #Seteamos parámetros para 
    ti = 0.5 #en segundos
    tf = 0.5 #en segundos
    descarteInicial = int(fm*ti) #en segundos
    descarteFinal = int(window*fm)-int(tf*fm) #en segundos

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
                    'shiftLen':window,
                    'ti': ti, 'tf':tf
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
                    'kernel_f': 4,
                    'n_ch': 1,
                    'num_classes': 4}

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData #la forma de joinedData es [estímulos, canales, muestras, trials]

    run1JoinedData = joinData(run1, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    run2JoinedData = joinData(run2, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)

    trainSet = np.concatenate((run1JoinedData[:,:,:,:12], run2JoinedData[:,:,:,:12]), axis = 3)
    trainSet = trainSet[:,:2, descarteInicial:descarteFinal,:] #nos quedamos con los primeros dos canales y descartamos muestras iniciales y algunas finales

    trainSet = np.mean(trainSet, axis = 1) #promedio sobre los canales. Forma datos ahora [clases, samples, trials]

    """
    **********************************************************************
    Second step: Create the CNN model
    **********************************************************************
    """

    nsamples = trainSet.shape[1]
    ntrials = trainSet.shape[2]

    #Restamos la media de la señal
    trainSet = trainSet - trainSet.mean(axis = 1, keepdims=True)

    #Make a CNNTrainingModule object in order to use the data's Magnitude Features
    cnn = CNNTrainingModule(trainSet, PRE_PROCES_PARAMS = PRE_PROCES_PARAMS, FFT_PARAMS = FFT_PARAMS, CNN_PARAMS = CNN_PARAMS,
                        frecStimulus = frecStimulus, nchannels = 1,nsamples = nsamples, ntrials = ntrials, modelName = "cnntesting")
    
    """
    **********************************************************************
    Third step: Compute and get the Features
    **********************************************************************
    """

    anchoVentana = (window - ti - tf) #fm * segundos
    ventana = windows.hamming

    sampleFrec, signalPSD  = cnn.featuresExtraction(ventana = ventana, anchoVentana = anchoVentana, bw = 2.0, order = 4, axis = 1)

    trainingData, labels = cnn.getDataForTraining(signalPSD)

    totalTrials = trainingData.shape[0]
    featuresPerTrial = trainingData.shape[1]
    inputshape = np.array([trainingData.shape[1], trainingData.shape[2], trainingData.shape[3]])


    """
    **********************************************************************
    Fourth step: Create the CNN model
    **********************************************************************
    """
    cnn.createModel(inputshape)


    """
    **********************************************************************
      Fifth step: Trainn the CNN
    **********************************************************************
    """
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder, "models")
    accu_CNN_using_MSF = cnn.trainCNN(trainingData, labels, nFolds = 5, path = path)
    print(f"Maxima accu {accu_CNN_using_MSF.max()}")

    #Guardamos modelo
    cnn.saveCNNModel(path, "models")
    cnn.saveTrainingSignalPSD(signalPSD.mean(axis = 2), path = path, filename = "cnntesting")

if __name__ == "__main__":
    main()
