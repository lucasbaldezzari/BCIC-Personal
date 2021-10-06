# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:06:02 2021

@author: Lucas Baldezzari

Clase que permite entrenar una SVM para clasificar SSVEPs a partir de datos de EEG.

************ VERSIÓN SCP-01-RevA ************
"""

import os
import numpy as np
import numpy.matlib as npm
import json

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import pickle

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import plotEEG
import fileAdmin as fa

class SVMTrainingModule():

    def __init__(self, rawEEG, subject, PRE_PROCES_PARAMS, FFT_PARAMS, modelName = ""):
        """Variables de configuración

        Args:
            - rawEEG(matrix[clases x canales x samples x trials]): Señal de EEG
            - subject (string o int): Número de sujeto o nombre de sujeto
            - PRE_PROCES_PARAMS: Parámetros para preprocesar los datos de EEG
            - FFT_PARAMS: Parametros para computar la FFT
            - modelName: Nombre del modelo
        """

        self.rawEEG = rawEEG
        self.subject = subject

        if not modelName:
            self.modelName = f"SVMModel_Subj{subject}"

        else:
            self.modelName = modelName

        self.eeg_channels = self.rawEEG.shape[0]
        self.total_trial_len = self.rawEEG.shape[2]
        self.num_trials = self.rawEEG.shape[3]

        self.model = None
        self.clases = None
        self.trainingData = None
        self.labels = None

        self.MSF = np.array([]) #Magnitud Spectrum Features

        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS

        self.METRICAS = {f'modelo_{self.modelName}': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                                                      'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

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
                                self.PRE_PROCES_PARAMS["bandStop"],
                                self.PRE_PROCES_PARAMS["sampling_rate"])

        #eeg data segmentation
        eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                      self.PRE_PROCES_PARAMS["shiftLen"],
                                      self.PRE_PROCES_PARAMS["sampling_rate"])

        self.MSF = computeMagnitudSpectrum(eegSegmented, self.FFT_PARAMS)

        return self.MSF


    #Transforming data for training
    def getDataForTraining(self, features, clases, canal = False):
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

        print("Generating training data")

        numFeatures = features.shape[0]
        canales = features.shape[1]
        numClases = features.shape[2]
        trials = features.shape[3]

        if canal == False:
            trainingData = np.mean(features, axis = 1)

        else:
            trainingData = features[:, canal, :, :]

        trainingData = trainingData.swapaxes(0,1).swapaxes(1,2).reshape(numClases*trials, numFeatures)

        classLabels = np.arange(len(clases))

        labels = (npm.repmat(classLabels, trials, 1).T).ravel()

        return trainingData, labels

    def createSVM(self, kernel, gamma, C):
        """Se crea modelo"""

        self.model = SVC(C = C, kernel = kernel, gamma = gamma)

        return self.model

    def trainAndValidateSVM(self, clases, test_size = 0.2):
        """Método para entrenar un modelo SVM.

        Argumentos:
            - clases (int): Lista con valores representando la cantidad de clases
            - test_size: Tamaño del set de validación"""

        self.clases = clases

        self.trainingData, self.labels = self.getDataForTraining(self.MSF, clases = self.clases)

        X_trn, X_val, y_trn, y_val = train_test_split(self.trainingData, self.labels, test_size = test_size)

        self.model.fit(X_trn,y_trn)

        y_pred = self.model.predict(X_trn)
        # accu = f1_score(y_val, y_pred, average='weighted')

        precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='weighted')

        self.METRICAS = {f'modelo_{self.modelName}': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                                                      'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

        precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='macro')
        accuracy = accuracy_score(y_trn, y_pred)


        self.METRICAS[f'modelo_{self.modelName}']['trn']['Pr'] = precision
        self.METRICAS[f'modelo_{self.modelName}']['trn']['Rc'] = recall
        self.METRICAS[f'modelo_{self.modelName}']['trn']['Acc'] = accuracy
        self.METRICAS[f'modelo_{self.modelName}']['trn']['F1'] = f1

        y_pred = self.model.predict(X_val)

        precision, recall, f1,_ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        accuracy = accuracy_score(y_val, y_pred)

        self.METRICAS[f'modelo_{self.modelName}']['val']['Pr'] = precision
        self.METRICAS[f'modelo_{self.modelName}']['val']['Rc'] = recall
        self.METRICAS[f'modelo_{self.modelName}']['val']['Acc'] = accuracy
        self.METRICAS[f'modelo_{self.modelName}']['val']['F1'] = f1

        return self.METRICAS

    def saveModel(self, path):
        """Método para guardar el modelo"""

        os.chdir(path)

        filename = f"{self.modelName}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

        #Guardamos los parámetros usados para entrenar el SVM
        file = open(f"{self.modelName}_preproces.json", "w")
        json.dump(self.PRE_PROCES_PARAMS , file)
        file.close

        file = open(f"{self.modelName}_fft.json", "w")
        json.dump(self.PRE_PROCES_PARAMS , file)
        file.close    

def main():

    """Empecemos"""

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG")

    frecStimulus = np.array([6, 8, 11])

    trials = 10
    fm = 200.
    window = 5 #sec
    samplePoints = int(fm*window)
    channels = 4
    stimuli = 1 #one stimulus

    subjects = [1] #un solo sujeto
    filenames = ["lucasB-R2-S1-E6","lucasB-R2-S1-E8", "lucasB-R3-S1-E11"]
    allData = fa.loadData(path = path, filenames = filenames)
    names = list(allData.keys())

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData

    joinedData = joinData(allData, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    #la forma de joinedData es [estímulos, canales, muestras, trials]

    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'lfrec': 5.,
                    'hfrec': 38.,
                    'order': 4,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': window,
                    'shiftLen':window
                    }

    resolution = np.round(fm/samplePoints, 4)

    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 5.0,
                    'end_frequency': 38.0,
                    'sampling_rate': fm
                    }

    #canales = 4

    trainSet = joinedData[:,:,:,:8] #me quedo con los primeros 8 trials para entrenamiento y validación

    #testSet = joinedData[:,:,:,8:] #me quedo con los últimos 2 trials para test

    svm1 = SVMTrainingModule(trainSet, "lucasB",PRE_PROCES_PARAMS,FFT_PARAMS, modelName = "SVM_LucasB_Test1_30092021")

    spectrum = svm1.computeMSF() #Computamos el espectro de Fourier de la señal

    modelo = svm1.createSVM(kernel = "linear", gamma = "scale", C = 1) #Creamos el modelo SVM

    metricas = svm1.trainAndValidateSVM(clases = np.arange(0,len(frecStimulus)), test_size = 0.2) #entrenamos el modelo

    print("**** METRICAS ****")
    print(metricas)

    #Chequeamos las features utilizadas para entrenar el SVM
    #Graficamos promediando sobre los trials de entrenamiento
    cantidadTrials = 8
    clase = 3
    fft_axis = np.arange(svm1.trainingData.shape[1]) * resolution
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [uV]')
    plt.title(f"Características para clase {frecStimulus[clase-1]} - Promedio sobre trials")
    plt.plot(fft_axis + FFT_PARAMS["start_frequency"],
              np.mean(svm1.trainingData[ (clase-1)*cantidadTrials : (clase-1)*cantidadTrials + cantidadTrials, :], axis = 0))
    plt.axvline(x = frecStimulus[clase-1], ymin = 0., ymax = max(fft_axis),
                          label = "Frecuencia estímulo",
                          linestyle='--', color = "#e37165", alpha = 0.9)
    plt.legend()
    plt.show()

    # Plotting para una clase y un trial
    cantidadTrials = 8
    trial = 0
    clase = 0
    fft_axis = np.arange(svm1.trainingData.shape[1]) * resolution
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [uV]')
    plt.title(f"Características para clase {frecStimulus[clase-1]} y trial {trial}")
    plt.plot(fft_axis + FFT_PARAMS["start_frequency"], svm1.trainingData[(clase-1)*cantidadTrials + (trial-1), :])
    plt.axvline(x = frecStimulus[clase-1], ymin = 0., ymax = max(fft_axis),
                          label = "Frecuencia estímulo",
                          linestyle='--', color = "#e37165", alpha = 0.9)

    plt.legend()
    plt.show()

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
    svm1.saveModel(path)
    os.chdir(actualFolder)

if __name__ == "__main__":
    main()

