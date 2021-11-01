"""
Created on Fri Sep 10 17:46:17 2021

@author: Lucas Baldezzari

LogRegClassifier: Clase que permite usar un clasificador
por Logistic Regression para clasificar SSVEPs a partir de datos de EEG

************ VERSIÓN SCP-01-RevA ************
"""

import os
import numpy as np
import numpy.matlib as npm
import pandas as pd
import json

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import pickle

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import plotEEG
from utils import norm_mean_std
import fileAdmin as fa

class LogRegClassifier():
    def __init__(self, modelFile, frecStimulus,
                    PRE_PROCES_PARAMS, FFT_PARAMS, nsamples, path = "models"):

        """Cosntructor de clase
        Argumentos:
            - modelFile: Nombre del archivo que contiene el modelo a cargar
            - frecStimulus: Lista con las frecuencias a clasificar
            - PRE_PROCES_PARAMS: Parámetros para preprocesar los datos de EEG
            - FFT_PARAMS: Parametros para computar la FFT
            -path: Carpeta donde esta guardado el modelo a cargar"""
        
        self.modelName = modelFile
        
        actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio featureVector
        os.chdir(path)
        
        with open(self.modelName, 'rb') as file:
            self.model = pickle.load(file)
            
        os.chdir(actualFolder)
        
        self.frecStimulus = frecStimulus
        self.nclases = len(frecStimulus)
        self.nsamples = nsamples
        
        self.rawDATA = None
        self.signalPSD = None
        self.signalSampleFrec = None
        self.signalPSDCentroid = None
        self.featureVector = None
        
        self.traingSigPSD = None
        self.trainSampleFrec = None
        self.trainPSDCent = []
        self.trainPSDDist = []

        #Setting variables for EEG processing.
        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS

    def loadTrainingSignalPSD(self, filename = "", path = "models"):

        actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
        os.chdir(path)

        if not filename:
            filename = f'{self.modelName}_signalPSD.txt'
        self.trainingSignalPSD = np.loadtxt(filename, delimiter=',')
        
        os.chdir(actualFolder)

    def applyFilterBank(self, eeg, bw = 2.0, order = 4):
        """Aplicamos banco de filtro a nuestros datos.
        Se recomienda aplicar un notch en los 50Hz y un pasabanda en las frecuencias deseadas antes
        de applyFilterBank()
        
        Args:
            - eeg: datos a aplicar el filtro. Forma [samples]
            - frecStimulus: lista con la frecuencia central de cada estímulo/clase
            - bw: ancho de banda desde la frecuencia central de cada estímulo/clase. Default = 2.0
            - order: orden del filtro. Default = 4"""

        nyquist = 0.5 * self.FFT_PARAMS["sampling_rate"]
        signalFilteredbyBank = np.zeros((self.nclases,self.nsamples))
        for clase, frecuencia in enumerate(self.frecStimulus):   
            low = (frecuencia-bw/2)/nyquist
            high = (frecuencia+bw/2)/nyquist
            b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
            signalFilteredbyBank[clase] = filtfilt(b, a, eeg) #filtramos

        self.dataBanked = signalFilteredbyBank.mean(axis = 0)

        return self.dataBanked

    def computWelchPSD(self, signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):

        self.signalSampleFrec, self.signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = axis)

        return self.signalSampleFrec, self.signalPSD

    def pearsonFilter(self):
        """Lo utilizamos para extraer nuestro vector de características en base a analizar la correlación entre nuestro
        banco de filtro entrenado y el banco de filtro obtenido a partir de datos de EEG nuevos"""

        """
                    |Var(X) Cov(X,Y)|
        cov(X,Y) =  |               |
                    |Cov(Y,X) Var(Y)|
        """
        
        r_pearson = []
        for clase, frecuencia in enumerate(self.frecStimulus):
            covarianza = np.cov(self.trainingSignalPSD[clase], self.signalPSD)
            r_i = covarianza/np.sqrt(covarianza[0][0]*covarianza[1][1])
            r_pearson.append(r_i[0][1])

        indexFfeature = r_pearson.index(max(r_pearson))  

        return self.trainingSignalPSD[indexFfeature]

    def extractFeatures(self, rawDATA, ventana, anchoVentana = 5, bw = 2.0, order = 4, axis = 1):

        filteredEEG = filterEEG(rawDATA, self.PRE_PROCES_PARAMS["lfrec"],
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

        self.featureVector = self.pearsonFilter()

        return self.featureVector

    def getClassification(self, featureVector):
        """Método para clasificar y obtener una frecuencia de estimulación a partir del EEG
        Argumentos:
            - rawEEG(matriz de flotantes [canales x samples]): Señal de EEG"""

        predicted = self.model.predict(featureVector.reshape(1, -1))
        
        return self.frecStimulus[predicted[0]]


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

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData #la forma de joinedData es [estímulos, canales, muestras, trials]

    run1JoinedData = joinData(run1, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    run2JoinedData = joinData(run2, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)

    testSet = np.concatenate((run1JoinedData[:,:,:,12:], run2JoinedData[:,:,:,12:]), axis = 3) #últimos 3 tríals para testeo
    testSet = testSet[:,:2,:,:] #nos quedamos con los primeros dos canales

    testSet = np.mean(testSet, axis = 1) #promedio sobre los canales. Forma datos ahora [clases, samples, trials]

    nsamples = testSet.shape[1]
    ntrials = testSet.shape[2]

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    
    path = os.path.join(actualFolder,"models")
    
    modelFile = "LogReg_WM_testing.pkl" #nombre del modelo

    logreg = LogRegClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, nsamples = nsamples, path = path) #cargamos clasificador entrenado
    logreg.loadTrainingSignalPSD(filename = "LogReg_WM_testing_signalPSD.txt", path = path) #cargamos el PSD de mis datos de entrenamiento

    clase = 4
    trial = 6

    rawDATA = testSet[clase-1,:,trial-1]

    featureVector = logreg.extractFeatures(rawDATA = rawDATA, ventana = windows.hamming, anchoVentana = 5, bw = 2.0, order = 4, axis = 0)

    print("Freceuncia clasificada:", logreg.getClassification(featureVector = featureVector))

    trials = 6
    predicciones = np.zeros((len(frecStimulus),trials))

    for i, clase in enumerate(np.arange(len(frecStimulus))):
        for j, trial in enumerate(np.arange(trials)):
            data = testSet[clase, :, trial]
            featureVector = logreg.extractFeatures(rawDATA = data, ventana = windows.hamming, anchoVentana = 5, bw = 2.0, order = 4, axis = 0)
            classification = logreg.getClassification(featureVector = featureVector)
            if classification == frecStimulus[clase]:
                predicciones[i,j] = 1

        predicciones[i,j] = predicciones[i,:].sum()/trials

    predictions = pd.DataFrame(predicciones, index = frecStimulus,
                    columns = [f"trial {trial+1}" for trial in np.arange(trials)])

    predictions['promedio'] = predictions.mean(numeric_only=True, axis=1)
    
    print(f"Predicciones usando el modelo SVM {modelFile}")
    print(predictions)