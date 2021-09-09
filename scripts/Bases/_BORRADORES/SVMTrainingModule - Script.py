# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:06:02 2021

@author: Lucas
"""

import os
import numpy as np
import numpy.matlib as npm
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import plotEEG
import fileAdmin as fa

def getDataForTraining(features, clases, canal = False):
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
                'bandStop': 50.,
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

magFFT = np.mean(magFFT, axis = 4) #media a través de los segmentos

trainingData, labels = getDataForTraining(magFFT, clases = np.arange(0,12))


#Checking the features
# Plotting promediando trials
cantidadTrials = 11
clase = 5
fft_axis = np.arange(trainingData.shape[1]) * resolution
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [uV]')
plt.title(f"Características para clase {frecStimulus[clase-1]} - Trials promediados")
plt.plot(fft_axis + FFT_PARAMS["start_frequency"],
          np.mean(trainingData[ (clase-1)*cantidadTrials : (clase-1)*cantidadTrials + cantidadTrials, :], axis = 0))
plt.axvline(x = frecStimulus[clase-1], ymin = 0., ymax = max(fft_axis),
                      label = "Frecuencia estímulo",
                      linestyle='--', color = "#e37165", alpha = 0.9)
plt.legend()
plt.show()

# Plotting para una clase y un trial
cantidadTrials = 11
trial = 11
clase = 5
fft_axis = np.arange(trainingData.shape[1]) * resolution
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [uV]')
plt.title(f"Características para clase {frecStimulus[clase-1]} y trial {trial}")
plt.plot(fft_axis + FFT_PARAMS["start_frequency"], trainingData[(clase-1)*cantidadTrials + (trial-1), :])
plt.axvline(x = frecStimulus[clase-1], ymin = 0., ymax = max(fft_axis),
                      label = "Frecuencia estímulo",
                      linestyle='--', color = "#e37165", alpha = 0.9)
plt.legend()
plt.show()

# Preparando datos de train y validation

X_trn, X_val, y_trn, y_val = train_test_split(trainingData, labels, test_size=.2)

hiperParams = {"kernels": ["linear", "rbf"],
    "gammaValues": [1e-2, 1e-1, 1, 1e+1, 1e+2, "scale", "auto"],
    "CValues": [8e-1,9e-1, 1, 1e2, 1e3]
    }

clasificadoresSVM = {"linear": list(),
                  "rbf": list()
    }

rbfResults = np.zeros((len(hiperParams["gammaValues"]), len(hiperParams["CValues"])))
linearResults = list()

for i, kernel in enumerate(hiperParams["kernels"]):
    
    if kernel != "linear":
        for j, gamma in enumerate(hiperParams["gammaValues"]):
            
            for k, C in enumerate(hiperParams["CValues"]):

                svm = SVC(C = C, kernel = kernel, gamma = gamma)
                svm.fit(X_trn,y_trn)
                y_pred = svm.predict(X_val)
                accu = f1_score(y_val, y_pred, average='weighted')
                
                rbfResults[j,k] = accu
                
                clasificadoresSVM[kernel].append((C, gamma, svm, accu))
    else:
        for k, C in enumerate(hiperParams["CValues"]):
            
            #Instanciamos el modelo para los hipermarametros
            model = SVC(C = C, kernel = kernel)
            #entreno el modelo
            model.fit(X_trn, y_trn)
            y_pred = model.predict(X_val)
            accu = f1_score(y_val, y_pred, average='weighted')
            linearResults.append(accu)
            
            clasificadoresSVM[kernel].append((C, model, accu))


plt.figure(figsize=(15,10))
plt.imshow(rbfResults)
plt.xlabel("Valor de C")
plt.xticks(np.arange(len(hiperParams["CValues"])), hiperParams["CValues"])
plt.ylabel("Valor de Gamma")
plt.yticks(np.arange(len(hiperParams["gammaValues"])), hiperParams["gammaValues"])
plt.colorbar();

for i in range(rbfResults.shape[0]):
    for j in range(rbfResults.shape[1]):
        plt.text(j, i, "{:.2f}".format(rbfResults[i, j]), va='center', ha='center')
plt.show()

num = np.arange(1,len(linearResults)+1)
plt.plot([str(C) for C in hiperParams["CValues"]], np.asarray(linearResults)*100)
plt.title("Accuracy para predicciones usando kernel 'linear'")
plt.xlabel("Valor de C") 
plt.ylabel("Accuracy (%)")
plt.show()

#probando algunos modelos
dataSetSegmentado = segmentingEEG(testSet, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                  PRE_PROCES_PARAMS["sampling_rate"])

magFFT = computeMagnitudSpectrum(dataSetSegmentado, FFT_PARAMS)
magFFT = np.mean(magFFT, axis = 4) #media a través de los segmentos

X_tst, y_tst = getDataForTraining(magFFT, clases = np.arange(0,12))


modeloSVM1 = clasificadoresSVM["linear"][2][1] #SVM con C=1 y kernel = "linear"

gamma = "scale"
C = 1.

for values in clasificadoresSVM["rbf"]:
    if values[0] == C and values[1] == gamma:
        modeloSVM2 = values[2] #modelo 2 es un SVM con kernel = rbf

modelos = {"SVM1": modeloSVM1,
            "SVM2": modeloSVM2}

generalResults = []


for modelo in modelos: # TODO completar con todos los modelos a usar
        
    modelos[modelo].fit(X_trn, y_trn)

    # La siguietne predicción se hace sobre datos que no fueron vistos antes
    pred = modelos[modelo].predict(X_tst)

    generalResults.append([modelo,
                            accuracy_score(y_tst, pred),
                            f1_score(y_tst, pred, average='weighted'),
                            recall_score(y_tst, pred, average='weighted'),
                            precision_score(y_tst, pred, average='weighted')])

generalResults = pd.DataFrame(generalResults, columns=["modelo", "acc", "f1", "pre", "rec"])
print(generalResults)


#Guardamos nuestros modelos
import pickle

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
os.chdir(path)

for modelo in modelos.keys():
    filename = f"{modelo}.pkl"
    with open(filename, 'wb') as file:  
        pickle.dump(modelos[modelo], file)

os.chdir(actualFolder)

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
os.chdir(path)

with open("SVM1.pkl", 'rb') as file:  
    SVM1 = pickle.load(file)

SVM1.predict(X_tst[4,:].reshape(1,-1))




