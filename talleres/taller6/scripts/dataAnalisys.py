# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:05:39 2021

@author: Lucas
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

prueba1 = allData[names[0]]
prueba2 = allData[names[1]]
prueba3 = allData[names[2]]

#Chequeamos información del registro prueba 2
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

# #NOTA IMPORTANTE: Los datos provenientes de la Synthetic board NO necesitan ser filtrados
# filteredEEG = filterEEG(eeg, PRE_PROCES_PARAMS["lfrec"],
#                         PRE_PROCES_PARAMS["hfrec"],
#                         PRE_PROCES_PARAMS["order"],
#                         PRE_PROCES_PARAMS["sampling_rate"])

prueba1EEGFiltered = filterEEG(prueba1EEG, PRE_PROCES_PARAMS["lfrec"],
                        PRE_PROCES_PARAMS["hfrec"],
                        PRE_PROCES_PARAMS["order"],
                        PRE_PROCES_PARAMS["sampling_rate"])

plotEEG(prueba1EEGFiltered, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")



#eeg data segmentation
eegSegmented = segmentingEEG(prueba1EEGFiltered, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

MSF = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)

canal = 6

plotOneSpectrum(MSF, resolution, 1, subjects[0], canal-1, [12.5],
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal  {canal}",
              folder = "figs")

"""Buscando SSVEPs"""

subjects = [1]

filenames = ["LucasB-PruebaSSVEPs(8Hz)-Num1","LucasB-PruebaSSVEPs-Num2","LucasB-PruebaSSVEPs-Num3"]
allData = fa.loadData(path = path, filenames = filenames)
names = list(allData.keys())

prueba1 = allData[names[0]]
prueba2 = allData[names[1]]
prueba3 = allData[names[2]]

#Chequeamos información del registro prueba 2
print(prueba1["generalInformation"])

prueba1EEG = prueba1["eeg"]#[:,:,:,1:] #descarto trial 1
#[Number of targets, Number of channels, Number of sampling points, Number of trials]

plotEEG(prueba1EEG, sujeto = 1, trial = 5, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

resolution = fm/prueba1EEG.shape[2]

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
                'end_frequency': 28.0,
                'sampling_rate': fm
                }

# #NOTA IMPORTANTE: Los datos provenientes de la Synthetic board NO necesitan ser filtrados
# filteredEEG = filterEEG(eeg, PRE_PROCES_PARAMS["lfrec"],
#                         PRE_PROCES_PARAMS["hfrec"],
#                         PRE_PROCES_PARAMS["order"],
#                         PRE_PROCES_PARAMS["sampling_rate"])

prueba1EEGFiltered = filterEEG(prueba1EEG, PRE_PROCES_PARAMS["lfrec"],
                        PRE_PROCES_PARAMS["hfrec"],
                        PRE_PROCES_PARAMS["order"],
                        PRE_PROCES_PARAMS["sampling_rate"])

plotEEG(prueba1EEGFiltered, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")



#eeg data segmentation
eegSegmented = segmentingEEG(prueba1EEGFiltered, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

MSF = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)

frec = [8]
canal = 1
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal} - Frecuencia {frec[0]}Hz",
              folder = "figs")

canal = 2
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal}",
              folder = "figs")

canal = 3
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal}",
              folder = "figs")

canal = 4
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal}",
              folder = "figs")

canal = 5
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal}",
              folder = "figs")

canal = 6
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal} - Frecuencia {frec[0]}Hz",
              folder = "figs")

canal = 7
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal} - Frecuencia {frec[0]}Hz",
              folder = "figs")

canal = 8
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal - 1, frec,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro para canal {canal} - Frecuencia {frec[0]}Hz",
              folder = "figs")
