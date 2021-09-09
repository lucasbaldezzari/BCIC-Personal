# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:48:15 2021

@author: Lucas
"""

import os
import numpy as np
import numpy.matlib as npm

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum, plotSpectrum
from utils import plotEEG
import fileAdmin as fa

#Transforming data for training
def getDataForTraining(features, clases):
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
    segmentos = features.shape[4]
    
    trainingData = features.swapaxes(0,4).swapaxes(0,1).swapaxes(3,1).swapaxes(2,3)
    trainingData = trainingData.reshape(canales*trials*segmentos*numClases, numFeatures)
    
    labels = np.repeat(clases, canales*trials*segmentos)

    return trainingData, labels

"""Let's starting"""
            
actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\talleres\\taller4\\scripts',"dataset")
# dataSet = sciio.loadmat(f"{path}/s{subject}.mat")

# path = "E:/reposBCICompetition/BCIC-Personal/taller4/scripts/dataset" #directorio donde estan los datos

subjects = np.arange(0,10)
# subjectsNames = [f"s{subject}" for subject in np.arange(1,11)]
subjectsNames = [f"s8"]

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
                'window': 4,
                'shiftLen':4
                }

resolution = fm/samples

FFT_PARAMS = {
                'resolution': resolution,#0.2930,
                'start_frequency': 5.0,
                'end_frequency': 38.0,
                'sampling_rate': fm
                }

for subject in subjectsNames:
    eeg = rawEEGs[subject]["eeg"]
    eeg = eeg[:,:, muestraDescarte: ,:]
    eeg = eeg[:,:, :tiempoTotal ,:]
    rawEEGs[subject]["eeg"] = filterEEG(eeg,lfrec = PRE_PROCES_PARAMS["lfrec"],
                                        hfrec = PRE_PROCES_PARAMS["hfrec"],
                                        orden = 4, bandStop = 50. , fm  = fm)
    
trainSet = rawEEGs["s8"]["eeg"][:,:,:,:11] #me quedo con los primeros 11 trials
testSet = rawEEGs["s8"]["eeg"][:,:,:,11:]

#eeg data segmentation
dataSetSegmentado = segmentingEEG(trainSet, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                  PRE_PROCES_PARAMS["sampling_rate"])

magFFT = computeMagnitudSpectrum(dataSetSegmentado, FFT_PARAMS)

# trainingData, labels = getDataForTraining(magFFT, clases = frecStimulus)
trainingData, labels = getDataForTraining(magFFT, clases = np.arange(0,12))

METRICAS = {'modelo': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                       'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                       'tst': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

X_trn, X_val, y_trn, y_val = train_test_split(trainingData, labels, test_size=.2)



# clasificador = LinearRegression()
# clasificador = LogisticRegression(solver = "newton-cg")
pipline = make_pipeline(StandardScaler(), LinearRegression())