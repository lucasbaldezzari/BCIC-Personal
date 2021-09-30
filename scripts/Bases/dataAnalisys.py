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

trials = 10
fm = 200.
window = 5 #sec
samplePoints = int(fm*window)
channels = 4
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["lucasB-R1-S1-E11","lucasB-R2-S1-E11","lucasB-R2-S1-E11"]
allData = fa.loadData(path = path, filenames = filenames)
names = list(allData.keys())

#Chequeamos información del registro prueba 1
print(allData[filenames[0]]["generalInformation"])

run1 = allData[names[0]]
run2 = allData[names[1]]
run3 = allData[names[2]]

#Chequeamos información del registro prueba 1
#print(prueba1["generalInformation"])
samples = run3["eeg"].shape[1]

run1EEG = run3["eeg"]
#[Number of targets, Number of channels, Number of sampling points, Number of trials]

plotEEG(run1EEG, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,5], rmvOffset = False, save = False, title = "", folder = "figs")

resolution = np.round(fm/run1EEG.shape[2], 4)

PRE_PROCES_PARAMS = {
                'lfrec': 5.,
                'hfrec': 18.,
                'order': 4,
                'sampling_rate': fm,
                'window': window,
                'shiftLen':window
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.,
                'end_frequency': 18.0,
                'sampling_rate': fm
                }

run1EEGFiltered = filterEEG(run1EEG, PRE_PROCES_PARAMS["lfrec"],
                        PRE_PROCES_PARAMS["hfrec"],
                        PRE_PROCES_PARAMS["order"],
                        PRE_PROCES_PARAMS["sampling_rate"])

plotEEG(run1EEGFiltered, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,5], rmvOffset = False, save = False,
            title = "Señal de EEG filtrada del Sujeto 1", folder = "figs")


#eeg data segmentation
eegSegmented = segmentingEEG(run1EEGFiltered, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

MSF = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
#(113, 8, 1, 3, 1)

canal = 4
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal-1, [11],
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = f"Espectro para canal  {canal}",
              folder = "figs")


# """Buscando SSVEPs"""
# subjects = [1] #cantidad de sujetos

# filenames = ["LucasB-PruebaSSVEPs(5.5Hz)-Num1",
#              "LucasB-PruebaSSVEPs(8Hz)-Num1",
#              "LucasB-PruebaSSVEPs(9Hz)-Num1"]
# allData = fa.loadData(path = path, filenames = filenames)
# names = list(allData.keys())

# estimuli = ["5.5","8","7"]
# frecStimulus = np.array([5.5, 8, 9])

# #Chequeamos información del registro de la prueba del estímulo 8hz
# print(allData["LucasB-PruebaSSVEPs(8Hz)-Num1"]["generalInformation"])

# for name in names:
#     print(f"Cantidad de trials para {name}:",
#           allData[name]["eeg"].shape[3])

# frec7hz = allData[names[0]]
# frec8hz = allData[names[1]]
# frec9hz = allData[names[2]]

# def joinData(allData, stimuli = 4, channels = 8, samples = 1000, trials = 15):
#     joinedData = np.zeros((stimuli, channels, samples, trials))
#     for i, sujeto in enumerate(allData):    
#         joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]
        
#     return joinedData

# joinedData = joinData(allData, stimuli = 3, channels = 8, samples = 1000, trials = 15)
# #la forma de joinedData es (3, 8, 1000, 15)[estímulos, canales, muestras, trials]

# #Graficamos el EEG de cada canal para cada estímulo
# trial = 10
# for stimulus in range(len(estimuli)):
#     plotEEG(joinedData, sujeto = 1, trial = 10, blanco = stimulus,
#             fm = fm, window = [0,4], rmvOffset = False, save = False,
#             title = f"EEG sin filtrar para target {estimuli[stimulus]}Hz",
#             folder = "figs")

# resolution = fm/joinedData.shape[2]

# PRE_PROCES_PARAMS = {
#                 'lfrec': 5.,
#                 'hfrec': 17.,
#                 'order': 4,
#                 'sampling_rate': fm,
#                 'window': 4,
#                 'shiftLen':4
#                 }

# FFT_PARAMS = {
#                 'resolution': resolution,
#                 'start_frequency': 0.0,
#                 'end_frequency': 17.0,
#                 'sampling_rate': fm
#                 }

# eegFiltered = filterEEG(joinedData, PRE_PROCES_PARAMS["lfrec"],
#                         PRE_PROCES_PARAMS["hfrec"],
#                         PRE_PROCES_PARAMS["order"],
#                         PRE_PROCES_PARAMS["sampling_rate"])

# #Graficamos el EEG de cada canal para cada estímulo
# trial = 10
# for stimulus in range(len(frecStimulus)):
#     plotEEG(eegFiltered, sujeto = 1, trial = 10, blanco = stimulus,
#             fm = fm, window = [0,4], rmvOffset = False, save = False,
#             title = f"EEG sin filtrar para target {estimuli[stimulus]}Hz",
#             folder = "figs")

# #eeg data segmentation
# eegSegmented = segmentingEEG(eegFiltered, PRE_PROCES_PARAMS["window"],
#                              PRE_PROCES_PARAMS["shiftLen"],
#                              PRE_PROCES_PARAMS["sampling_rate"])

# magnitudFeatures = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
# #MSF.shape = [features, canales, estímulos, trials, segmentos]
# #(113, 8, 1, 3, 1)
# cantidadTargets = 3
# plotSpectrum(magnitudFeatures, resolution, cantidadTargets,
#              subjects[0], 7, frecStimulus,
#               startFrecGraph = FFT_PARAMS['start_frequency'],
#               save = False, title = "", folder = "figs",
#               rows = 1, columns = 3)