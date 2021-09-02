# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:05:39 2021

@author: Lucas

        VERSIÓN: SCT-01-RevA
"""
import os
import numpy as np

import fileAdmin as fa
from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum, plotEEG

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG")

trials = 15
fm = 250.
window = 4 #sec
samplePoints = int(fm*window)
channels = 8
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["LucasB-Alfa-Prueba1","LucasB-Alfa-Prueba2","LucasB-Alfa-Prueba3"]
allData = fa.loadData(path = path, filenames = filenames)
names = list(allData.keys())

#Chequeamos información del registro prueba 1
print(allData["LucasB-Alfa-Prueba1"]["generalInformation"])

prueba1 = allData[names[0]]
prueba2 = allData[names[1]]
prueba3 = allData[names[2]]

#Chequeamos información del registro prueba 1
print(prueba1["generalInformation"])

prueba1EEG = prueba3["eeg"][:,:,:,1:] #descarto trial 1
#[Number of targets, Number of channels, Number of sampling points, Number of trials]

plotEEG(prueba1EEG, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

resolution = fm/prueba1EEG.shape[2]

PRE_PROCES_PARAMS = {
                'lfrec': 7.,
                'hfrec': 28.,
                'order': 4,
                'sampling_rate': fm,
                'window': 4,
                'shiftLen':4
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.0,
                'end_frequency': 28.0,
                'sampling_rate': fm
                }

prueba1EEGFiltered = filterEEG(prueba1EEG, PRE_PROCES_PARAMS["lfrec"],
                        PRE_PROCES_PARAMS["hfrec"],
                        PRE_PROCES_PARAMS["order"],
                        PRE_PROCES_PARAMS["sampling_rate"])

plotEEG(prueba1EEGFiltered, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False,
            title = "Señal de EEG filtrada del Sujeto 1", folder = "figs")


#eeg data segmentation
eegSegmented = segmentingEEG(prueba1EEGFiltered, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

MSF = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
#(113, 8, 1, 3, 1)

canal = 6
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal-1, [12.5],
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = f"Espectro para canal  {canal}",
              folder = "figs")


"""Buscando SSVEPs"""
subjects = [1] #cantidad de sujetos

filenames = ["LucasB-PruebaSSVEPs(5.5Hz)-Num1",
             "LucasB-PruebaSSVEPs(8Hz)-Num1",
             "LucasB-PruebaSSVEPs(9Hz)-Num1"]
allData = fa.loadData(path = path, filenames = filenames)
names = list(allData.keys())

estimuli = ["5.5","8","7"]
frecStimulus = np.array([5.5, 8, 9])

#Chequeamos información del registro de la prueba del estímulo 8hz
print(allData["LucasB-PruebaSSVEPs(8Hz)-Num1"]["generalInformation"])

for name in names:
    print(f"Cantidad de trials para {name}:",
          allData[name]["eeg"].shape[3])

frec7hz = allData[names[0]]
frec8hz = allData[names[1]]
frec9hz = allData[names[2]]

def joinData(allData, stimuli = 4, channels = 8, samples = 1000, trials = 15):
    joinedData = np.zeros((stimuli, channels, samples, trials))
    for i, sujeto in enumerate(allData):    
        joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]
        
    return joinedData

joinedData = joinData(allData, stimuli = 3, channels = 8, samples = 1000, trials = 15)
#la forma de joinedData es (3, 8, 1000, 15)[estímulos, canales, muestras, trials]

#Graficamos el EEG de cada canal para cada estímulo
trial = 10
for stimulus in range(len(estimuli)):
    plotEEG(joinedData, sujeto = 1, trial = 10, blanco = stimulus,
            fm = fm, window = [0,4], rmvOffset = False, save = False,
            title = f"EEG sin filtrar para target {estimuli[stimulus]}Hz",
            folder = "figs")

resolution = fm/joinedData.shape[2]

PRE_PROCES_PARAMS = {
                'lfrec': 5.,
                'hfrec': 17.,
                'order': 4,
                'sampling_rate': fm,
                'window': 4,
                'shiftLen':4
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.0,
                'end_frequency': 17.0,
                'sampling_rate': fm
                }

eegFiltered = filterEEG(joinedData, PRE_PROCES_PARAMS["lfrec"],
                        PRE_PROCES_PARAMS["hfrec"],
                        PRE_PROCES_PARAMS["order"],
                        PRE_PROCES_PARAMS["sampling_rate"])

#Graficamos el EEG de cada canal para cada estímulo
trial = 10
for stimulus in range(len(frecStimulus)):
    plotEEG(eegFiltered, sujeto = 1, trial = 10, blanco = stimulus,
            fm = fm, window = [0,4], rmvOffset = False, save = False,
            title = f"EEG sin filtrar para target {estimuli[stimulus]}Hz",
            folder = "figs")

#eeg data segmentation
eegSegmented = segmentingEEG(eegFiltered, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

magnitudFeatures = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
#MSF.shape = [features, canales, estímulos, trials, segmentos]
#(113, 8, 1, 3, 1)
cantidadTargets = 3
plotSpectrum(magnitudFeatures, resolution, cantidadTargets,
             subjects[0], 7, frecStimulus,
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = "", folder = "figs",
              rows = 1, columns = 3)
