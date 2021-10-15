# -*- coding: utf-8 -*-
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

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import pickle

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import plotEEG
from utils import norm_mean_std
import fileAdmin as fa

class LogRegClassifier():
    
    def __init__(self, modelFile, frecStimulus,
                 PRE_PROCES_PARAMS, FFT_PARAMS, path = "models"):
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
            self.logreg = pickle.load(file)
            
        os.chdir(actualFolder)
        
        self.frecStimulusList = frecStimulus #clases
        
        self.rawEEG = None
        
        self.MSF = np.array([]) #Magnitud Spectrum Features
        
        #Setting variables for EEG processing.
        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS
        
    def reshapeRawEEG(self, rawEEG):
        """Transformamos los datos de EEG en la forma adecuada para poder procesar la FFT y obtener el espectro
        
        #Es importante tener en cuenta que los datos de OpenBCI vienen en la forma [canales x samples] y
        el método computeMagnitudSpectrum() esta preparado para computar el espectro con los datos de la forma
        [clases x canales x samples x trials]
        
        """
        
        numCanales = rawEEG.shape[0]
        numFeatures = rawEEG.shape[1]
        self.rawEEG = rawEEG.reshape(1, numCanales, numFeatures, 1)
        
        return self.rawEEG 
        
    def computeMSF(self):
        """Compute the FFT over segmented EEG data.
        
        Argument: None. This method use variables from the own class
        
        Return: The Magnitud Spectrum Feature (MSF)."""
        
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
    
    #Transofrmamos los datos del magnitud spectrum features en un formato para la SVM
    def transformDataForClassifier(self, features, canal = False):
        """Preparación del set de entrenamiento.
            
        Argumentos:
            - features: Parte Real del Espectro or Parte Real e Imaginaria del Espectro
            con forma [número de características x canales x clases x trials x número de segmentos]
            - clases: Lista con las clases para formar las labels
            
        Retorna:
            - dataForSVM: Set de datos de entrenamiento para alimentar el modelo SVM
            Con forma [trials*clases x number of features]
            - Labels: labels para entrenar el modelo a partir de las clases"""
        
        #print("Transformando datos para clasificarlos")
        
        numFeatures = features.shape[0]
        canales = features.shape[1]
        numClases = features.shape[2]
        trials = features.shape[3]
        
        if canal == False:
            dataForSVM = np.mean(features, axis = 1)
            
        else:
            dataForSVM = features[:, canal, :, :]
            
        dataForSVM = dataForSVM.swapaxes(0,1).swapaxes(1,2).reshape(numClases*trials, numFeatures)
        
        return dataForSVM
    
    def getClassification(self, rawEEG):
        """Método para clasificar y obtener una frecuencia de estimulación a partir del EEG
        Argumentos:
            - rawEEG(matriz de flotantes [canales x samples]): Señal de EEG"""
        
        reshapedEEG = self.reshapeRawEEG(rawEEG) #transformamos los datos originales
        
        rawFeatures = self.computeMSF() #computamos la FFT para extraer las características
        
        dataForSVM = self.transformDataForClassifier(rawFeatures) #transformamos el espacio de características
        
        index = self.logreg.predict(dataForSVM)[0] #clasificamos
        
        return self.frecStimulusList[index] #retornamos la frecuencia clasificada
    
    
def main():
    """Let's starting"""
                    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG\\LucasB")

    frecStimulus = np.array([7, 9, 11, 13])

    trials = 15
    fm = 200.
    window = 5 #sec
    samplePoints = int(fm*window)
    channels = 4

    filesRun1 = ["lb-R1-S1-E7","lb-R1-S1-E9", "lb-R1-S1-E11","lb-R1-S1-E13"]
    run1 = fa.loadData(path = path, filenames = filesRun1)
    filesRun2 = ["lb-R2-S1-E7","lb-R2-S1-E9", "lb-R2-S1-E11","lb-R2-S1-E13"]
    run2 = fa.loadData(path = path, filenames = filesRun2)

    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'lfrec': 5.,
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
                    'start_frequency': 5.0,
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

    testSet = np.concatenate((run1JoinedData[:,:,:,12:], run2JoinedData[:,:,:,12:]), axis = 3)
    testSet = testSet[:,:2,:,:] #nos quedamos con los primeros dos canales
    #testSet = norm_mean_std(testSet)
    
    #testSet = joinedData[:,:,:,12:] #me quedo con los últimos 2 trials para test
    #testSet = testSet[:,:2,:,:] #nos quedamos con los primeros dos canales
    
    path = "E:\reposBCICompetition\BCIC-Personal\scripts\Bases\models"
    
    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
    
    modelFile = "Logreg_LucasB_Test2_10112021.pkl"
        
    logreg = LogRegClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, path = path)
    
    #De nuestro set de datos seleccionamos el EEG de correspondiente a una clase y un trial.
    #Es importante tener en cuenta que los datos de OpenBCI vienen en la forma [canales x samples]
    
    clase = 1 #corresponde al estímulo de 7Hz
    trial = 6
    
    rawEEG = testSet[clase - 1, :, : , trial - 1]
    
    frecClasificada = logreg.getClassification(rawEEG = rawEEG)
    print(f"El estímulo clasificado fue {frecClasificada}")
    
    clase = 3 #corresponde al estímulo de 11Hz
    trial = 3
    
    rawEEG = testSet[clase - 1, :, : , trial - 1]
    
    frecClasificada = logreg.getClassification(rawEEG = rawEEG)
    print(f"El estímulo clasificado fue {frecClasificada}")

    trials = 6
    predicciones = np.zeros((len(frecStimulus),trials))
    
    for i, clase in enumerate(np.arange(4)):
        for j, trial in enumerate(np.arange(6)):
            data = testSet[clase, :, : , trial]
            classification = logreg.getClassification(rawEEG = data)
            if classification == frecStimulus[clase]:
                predicciones[i,j] = 1

        #predicciones[i,j+1] = predicciones[i,:].sum()/trials

    predictions = pd.DataFrame(predicciones, index = frecStimulus,
                    columns = [f"trial {trial+1}" for trial in np.arange(trials)])

    predictions['promedio'] = predictions.mean(numeric_only=True, axis=1)
    
    print(f"Predicciones usando el modelo LogReg {modelFile}")
    print(predictions)


if __name__ == "__main__":
    main()

