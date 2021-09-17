# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:12:08 2021

@author: Lucas Baldezzari

SVMClassifier: Clase que permiteusar un SVM para clasificar SSVEPs a partir de datos de EEG

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
import fileAdmin as fa

class SVMClassifier():
    
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
            self.svm = pickle.load(file)
            
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
        
        print("Transformando datos para clasificarlos")
        
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
        
        index = self.svm.predict(dataForSVM)[0] #clasificamos
        
        return self.frecStimulusList[index] #retornamos la frecuencia clasificada
    
    
def main():
    """Let's starting"""
                    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\talleres\\taller4\\scripts',"dataset")
    
    subjects = np.arange(0,10)
    # subjectsNames = [f"s{subject}" for subject in np.arange(1,11)]
    subjectsNames = ["s8"]
    
    fm = 256.0
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
    
    """Loading the EEG data"""
    rawEEGs = fa.loadData(path = path, filenames = subjectsNames)
    
    
    samples = rawEEGs[subjectsNames[0]]["eeg"].shape[2] #the are the same for all sobjecs and trials
    
    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'lfrec': 5.,
                    'hfrec': 38.,
                    'order': 4,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': 4,
                    'shiftLen':4
                    }
    
    resolution = np.round(fm/tiempoTotal, 4)
    
    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 5.0,
                    'end_frequency': 38.0,
                    'sampling_rate': fm
                    }
    
    # for subject in subjectsNames:
    #     eeg = rawEEGs[subject]["eeg"]
    #     eeg = eeg[:,:, muestraDescarte: ,:]
    #     eeg = eeg[:,:, :tiempoTotal ,:]
    #     rawEEGs[subject]["eeg"] = filterEEG(eeg,lfrec = PRE_PROCES_PARAMS["lfrec"],
    #                                         hfrec = PRE_PROCES_PARAMS["hfrec"],
    #                                         orden = 4, bandStop = 50. , fm  = fm)

    canales = 4
    rawEEGs["s8"]["eeg"] = rawEEGs["s8"]["eeg"][:,:, muestraDescarte: ,:]    
    rawEEGs["s8"]["eeg"] = rawEEGs["s8"]["eeg"][:,:, :tiempoTotal ,:]
    rawEEGs["s8"]["eeg"] = rawEEGs["s8"]["eeg"][:, 0:canales, :, :]
        
    testSet = rawEEGs["s8"]["eeg"][:,:,:,11:] #seleccionamos los últimos 4 trials
    
    path = "E:\reposBCICompetition\BCIC-Personal\scripts\Bases\models"
    
    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
    
    modelFile = "SVM14Channels.pkl"
        
    svm = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, path = path)
    
    #De nuestro set de datos seleccionamos el EEG de correspondiente a una clase y un trial.
    #Es importante tener en cuenta que los datos de OpenBCI vienen en la forma [canales x samples]
    
    clase = 5 #corresponde al estímulo de 11.75Hz
    trial = 2
    
    rawEEG = testSet[clase - 1, :, : , trial - 1]
    
    frecClasificada = svm.getClassification(rawEEG = rawEEG)
    print(f"El estímulo clasificado fue {frecClasificada}")
    
    clase = 9 #corresponde al estímulo de 14.25Hz
    trial = 2
    
    rawEEG = testSet[clase - 1, :, : , trial - 1]
    
    frecClasificada = svm.getClassification(rawEEG = rawEEG)
    print(f"El estímulo clasificado fue {frecClasificada}")


if __name__ == "__main__":
    main()