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

import pickle

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import norm_mean_std

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
        
        self.rawDATA = None
        
        self.signalPSD = np.array([]) #Magnitud Spectrum Features
        
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
        self.rawDATA = rawEEG.reshape(1, numCanales, numFeatures, 1)
        
        return self.rawDATA 
        
    def computeMSF(self):
        """Compute the FFT over segmented EEG data.
        
        Argument: None. This method use variables from the own class
        
        Return: The Magnitud Spectrum Feature (MSF)."""
        
        #eeg data filtering
        filteredEEG = filterEEG(self.rawDATA, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["bandStop"],
                                self.PRE_PROCES_PARAMS["sampling_rate"])
        
        #eeg data segmentation
        eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                      self.PRE_PROCES_PARAMS["shiftLen"],
                                      self.PRE_PROCES_PARAMS["sampling_rate"])
        
        self.signalPSD = computeMagnitudSpectrum(eegSegmented, self.FFT_PARAMS)
        
        return self.signalPSD
    
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
        
        index = self.svm.predict(dataForSVM)[0] #clasificamos
        
        return self.frecStimulusList[index] #retornamos la frecuencia clasificada
    
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

    testSet = np.concatenate((run1JoinedData[:,:,:,12:], run2JoinedData[:,:,:,12:]), axis = 3)
    testSet = testSet[:,:2,:,:] #nos quedamos con los primeros dos canales
    #testSet = norm_mean_std(testSet) #normalizamos los datos

    #trainSet = joinedData[:,:,:,:12] #me quedo con los primeros 12 trials para entrenamiento y validación
    #trainSet = trainSet[:,:2,:,:] #nos quedamos con los primeros dos canales
    
    path = "E:\reposBCICompetition\BCIC-Personal\scripts\Bases\models"
    
    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
    
    #modelFile = "SVM_LucasB_100accu_14102021.pkl" #nombre del modelo
    modelFile = "SVM_WM_2chann_rojo_rbf_221021.pkl" #nombre del modelo
        
    svm = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, path = path)
    
    #De nuestro set de datos seleccionamos el EEG de correspondiente a una clase y un trial.
    #Es importante tener en cuenta que los datos de OpenBCI vienen en la forma [canales x samples]
    
    clase = 1 #corresponde al estímulo de 6Hz
    trial = 2
    
    rawEEG = testSet[clase - 1, :, : , trial - 1]
    
    frecClasificada = svm.getClassification(rawEEG = rawEEG)
    print(f"El estímulo clasificado fue {frecClasificada}")
    
    clase = 4
    trial = 3
    
    rawEEG = testSet[clase - 1, :, : , trial - 1]
    
    frecClasificada = svm.getClassification(rawEEG = rawEEG)
    print(f"El estímulo clasificado fue {frecClasificada}")

    trials = 6
    predicciones = np.zeros((len(frecStimulus),trials))
    
    for i, clase in enumerate(np.arange(4)):
        for j, trial in enumerate(np.arange(6)):
            data = testSet[clase, :, : , trial]
            classification = svm.getClassification(rawEEG = data)
            if classification == frecStimulus[clase]:
                predicciones[i,j] = 1

        #predicciones[i,j+1] = predicciones[i,:].sum()/trials

    predictions = pd.DataFrame(predicciones, index = frecStimulus,
                    columns = [f"trial {trial+1}" for trial in np.arange(trials)])

    predictions['promedio'] = predictions.mean(numeric_only=True, axis=1)
    
    print(f"Predicciones usando el modelo SVM {modelFile}")
    print(predictions)

if __name__ == "__main__":
    main()