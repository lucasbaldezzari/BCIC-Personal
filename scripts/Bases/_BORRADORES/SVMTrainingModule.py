# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 19:41:50 2021

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

X_trn, X_val, y_trn, y_val = train_test_split(trainingData, labels, test_size=.2)

#eeg data segmentation
testSetSegmentado = segmentingEEG(testSet, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                  PRE_PROCES_PARAMS["sampling_rate"])

magFFT = computeMagnitudSpectrum(testSetSegmentado, FFT_PARAMS)

# trainingData, labels = getDataForTraining(magFFT, clases = frecStimulus)
X_test, y_test = getDataForTraining(magFFT, clases = np.arange(0,12))

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X_trn,y_trn)
# y_pred = clf.predict(X_val)
# accu = f1_score(y_val, y_pred, average='micro')


hiperParams = {"kernels": ["linear", "rbf", "poly", "sigmoid"],
    "gammaValues": [1e-2, 1e-1, 1, 1e+1, 1e+2, "scale", "auto"],
    "CValues": [8e-1,9e-1, 1, 1e2, 1e3]
    }


clasificadoresSVM = {"linear": list(),
                  "rbf": list(),
                  "poly": list(),
                  "sigmoid": list()
    }

rbfResults = np.zeros((len(hiperParams["gammaValues"]), len(hiperParams["CValues"])))
linearResults = list()

for i, kernel in enumerate(hiperParams["kernels"]):
    
    if kernel != "linear":
        for j, gamma in enumerate(hiperParams["gammaValues"]):
            
            for k, C in enumerate(hiperParams["CValues"]):
                #Instanciamos el modelo para los hipermarametros
                model = SVC(C = C, kernel = kernel, gamma = gamma)
                
                #entreno el modelo
                model.fit(X_trn, y_trn)
                
                #predecimos con los datos en Xoptim
                y_pred = model.predict(X_val)
                accu = f1_score(y_val, y_pred, average='micro')
                
                rbfResults[j,k] = accu
                
                clasificadoresSVM[kernel].append((C, gamma, model, accu))
    else:
        for k, C in enumerate(hiperParams["CValues"]):
            
            #Instanciamos el modelo para los hipermarametros
            model = SVC(C = C, kernel = kernel)
            #entreno el modelo
            model.fit(X_trn, y_trn)
            y_pred = model.predict(X_val)
            accu = f1_score(y_val, y_pred, average='micro')
            linearResults.append(accu)
            #predecimos con los datos en Xoptim
            
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

##PCA

components = 4
clases = trainSet.shape[0]
channels = trainSet.shape[1]
samples = trainSet.shape[2]
trials = trainSet.shape[3]

trainSetPCA = np.zeros((clases, components, samples, trials))


for clase in range(len(frecStimulus)):
    pca = PCA(n_components = components)

    data = trainSet[clase,:,:,:].reshape(channels, samples*trials).swapaxes(0,1)
    pca.fit(data)
    trainSetPCA[clase] = pca.transform(data).swapaxes(0,1).reshape(components,samples,trials)
    
components = 4
clases = testSet.shape[0]
channels = testSet.shape[1]
samples = testSet.shape[2]
trials = testSet.shape[3]

testSetPCA = np.zeros((clases, components, samples, trials))


for clase in range(len(frecStimulus)):
    pca = PCA(n_components = components)

    data = testSet[clase,:,:,:].reshape(channels, samples*trials).swapaxes(0,1)
    pca.fit(data)
    testSetPCA[clase] = pca.transform(data).swapaxes(0,1).reshape(components,samples,trials)
    
dataSetSegmentado = segmentingEEG(trainSetPCA, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                  PRE_PROCES_PARAMS["sampling_rate"])

magFFT = computeMagnitudSpectrum(dataSetSegmentado, FFT_PARAMS)

# trainingData, labels = getDataForTraining(magFFT, clases = frecStimulus)
trainingDataPCA, labelsPCA = getDataForTraining(magFFT, clases = np.arange(0,12))

X_trnPCA, X_valPCA, y_trnPCA, y_valPCA = train_test_split(trainingDataPCA, labelsPCA, test_size=.2)

#eeg data segmentation
testSetSegmentado = segmentingEEG(testSetPCA, PRE_PROCES_PARAMS["window"],PRE_PROCES_PARAMS["shiftLen"],
                                  PRE_PROCES_PARAMS["sampling_rate"])

magFFT = computeMagnitudSpectrum(testSetSegmentado, FFT_PARAMS)

# trainingData, labels = getDataForTraining(magFFT, clases = frecStimulus)
X_testPCA, y_testPCA = getDataForTraining(magFFT, clases = np.arange(0,12))

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

                clf = make_pipeline(StandardScaler(), SVC(C = C, kernel = kernel, gamma = gamma))
                clf.fit(X_trnPCA,y_trnPCA)
                y_pred = clf.predict(X_valPCA)
                accu = f1_score(y_valPCA, y_pred, average='micro')
                
                rbfResults[j,k] = accu
                
                clasificadoresSVM[kernel].append((C, gamma, model, accu))
    else:
        for k, C in enumerate(hiperParams["CValues"]):
            
            #Instanciamos el modelo para los hipermarametros
            model = SVC(C = C, kernel = kernel)
            #entreno el modelo
            model.fit(X_trnPCA, y_trnPCA)
            y_pred = model.predict(X_valPCA)
            accu = f1_score(y_valPCA, y_pred, average='micro')
            linearResults.append(accu)
            #predecimos con los datos en Xoptim
            
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
