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
filenames = ["lucasB-R2-S1-E7","lucasB-R2-S1-E13","lucasB-R2-S1-E11"]
allData = fa.loadData(path = path, filenames = filenames)
names = list(allData.keys())

#Chequeamos información del registro prueba 1
print(allData[filenames[0]]["generalInformation"])

run1 = allData[names[0]]
run2 = allData[names[1]]
run3 = allData[names[2]]

#Chequeamos información del registro prueba 1
print(run2["generalInformation"])

run1EEG = run2["eeg"]
#[Number of targets, Number of channels, Number of sampling points, Number of trials]

plotEEG(run1EEG, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,5], rmvOffset = False, save = False, title = "", folder = "figs")

resolution = np.round(fm/run1EEG.shape[2], 4)

PRE_PROCES_PARAMS = {
                'lfrec': 5.,
                'hfrec': 28.,
                'order': 8,
                'sampling_rate': fm,
                'window': window,
                'shiftLen':window
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.,
                'end_frequency': 28.0,
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

canal = 1
estim = [13]
plotOneSpectrum(MSF, resolution, 1, subjects[0], canal-1, estim,
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False, title = f"Espectro para canal  {canal}",
              folder = "figs")

#Graficamos espectro promediando canales
import matplotlib.pyplot as plt
trial = 10
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [uV]')
trial = 3
plt.title(f"Espectro promediando los 4 para trial {trial} y estim {estim[0]}")
fft_axis = np.arange(MSF.shape[0]) * resolution
MSFmean = np.mean(MSF,axis = 1).reshape(MSF.shape[0],MSF.shape[3])
plt.plot(fft_axis, MSFmean[:,trial-1])

plt.axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
                        label = f"Frecuencia estímulo {estim[0]}Hz",
                        linestyle='--', color = "#e37165", alpha = 0.9)
plt.legend()
plt.show()

#Graficamos espectro promediando canales
import matplotlib.pyplot as plt
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [uV]')
plt.title(f"Espectro promediando los 4 y todos los trials - {estim[0]}")
fft_axis = np.arange(MSF.shape[0]) * resolution
MSFmean = np.mean(MSF,axis = 1).reshape(MSF.shape[0],MSF.shape[3])
plt.plot(fft_axis, np.mean(MSFmean, axis = 1))

plt.axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
                        label = f"Frecuencia estímulo {estim[0]}Hz",
                        linestyle='--', color = "#e37165", alpha = 0.9)
plt.legend()
plt.show()