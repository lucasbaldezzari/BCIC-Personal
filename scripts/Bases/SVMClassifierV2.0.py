
import os
import numpy as np
import numpy.matlib as npm
import pandas as pd

import pickle

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import norm_mean_std

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

import fileAdmin as fa

class SVMClassifier():
    
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
        
        actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
        os.chdir(path)
        
        with open(self.modelName, 'rb') as file:
            self.svm = pickle.load(file)
            
        os.chdir(actualFolder)
        
        self.frecStimulus = frecStimulus
        self.nclases = len(frecStimulus)
        self.nsamples = nsamples
        
        self.rawDATA = None
        self.signalPSD = None
        
        self.trainingSignalPSD = None
        
        #Setting variables for EEG processing.
        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS
        
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

        print(r_pearson)
        indexFfeature = r_pearson.index(max(r_pearson))  
        print(self.frecStimulus[indexFfeature])

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

        return self.signalSampleFrec, self.signalPSD

    def loadTrainingSignalPSD(self, filename = "", path = "models"):

        actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
        os.chdir(path)

        if not filename:
            filename = f'{self.modelName}_signalPSD.txt'
        self.trainingSignalPSD = np.loadtxt(filename, delimiter=',')
        
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

    trainSet = np.concatenate((run1JoinedData[:,:,:,12:], run2JoinedData[:,:,:,12:]), axis = 3) #últimos 3 tríals para testeo
    trainSet = trainSet[:,:2,:,:] #nos quedamos con los primeros dos canales

    trainSet = np.mean(trainSet, axis = 1) #promedio sobre los canales. Forma datos ahora [clases, samples, trials]

    nsamples = trainSet.shape[1]
    ntrials = trainSet.shape[2]

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    
    path = os.path.join(actualFolder,"models")
    
    modelFile = "SVM_test_rbf.pkl" #nombre del modelo

    svm = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, nsamples = nsamples, path = path) #cargamos clasificador entrenado
    svm.loadTrainingSignalPSD(filename = "SVM_test_rbf_signalPSD.txt", path = path) #cargamos el PSD de mis datos de entrenamiento

    rawDATA = trainSet[2,:,5]

    filteredEEG = filterEEG(rawDATA, svm.PRE_PROCES_PARAMS["lfrec"],
                            svm.PRE_PROCES_PARAMS["hfrec"],
                            svm.PRE_PROCES_PARAMS["order"],
                            svm.PRE_PROCES_PARAMS["bandStop"],
                            svm.PRE_PROCES_PARAMS["sampling_rate"],
                            axis = 0)

    dataBanked = svm.applyFilterBank(filteredEEG, bw=1, order = 8)

    anchoVentana = int(svm.PRE_PROCES_PARAMS["sampling_rate"]*5) #fm * segundos
    ventana = windows.hamming(anchoVentana)

    signalSampleFrec, signalPSD = svm.computWelchPSD(dataBanked,
                                            fm = svm.PRE_PROCES_PARAMS["sampling_rate"],
                                            ventana = ventana, anchoVentana = anchoVentana,
                                            average = "median", axis = 0)

    svm.extractFeatures(rawDATA = rawDATA, ventana = windows.hamming, anchoVentana = 5, bw = 2.0, order = 4, axis = 0)
    featureVector = svm.pearsonFilter()
    # frecClasificada = svm.getClassification(rawDATA = rawDATA)

if __name__ == "__main__":
    main()

