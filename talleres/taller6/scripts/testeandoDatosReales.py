# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:17:27 2021

@author: Lucas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from utils import filterEEG, plotEEG, segmentingEEG, computeMagnitudSpectrum, plotSpectrum, plotOneSpectrum

initialFolder = os.getcwd() #directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(initialFolder,"recordedEEG")

data1 = pd.read_csv(f'{path}\prueba1.txt', delimiter = ",")
data2 = pd.read_csv(f'{path}\prueba2.txt', delimiter = ",")

canal1Data1Name = data1.columns[1]
canal2Data1Name = data1.columns[2]
canal3Data1Name = data1.columns[3]
canal4Data1Name = data1.columns[4]
canal5Data1Name = data1.columns[5]
canal6Data1Name = data1.columns[6]
canal7Data1Name = data1.columns[7]
canal8Data1Name = data1.columns[8]

canal1Data2Name = data2.columns[1]
canal2Data2Name = data2.columns[2]



# eeg1 = np.array((np.asarray(data1[canal1Data1Name]),np.asarray(data1[canal2Data1Name])))
# eeg2 = np.array((np.asarray(data2[canal1Data2Name]),np.asarray(data2[canal2Data2Name])))


# datosCanla1 = filterEEG(eeg1, lfrec = 5., hfrec = 38., orden = 4, fm  = 250.0)

fm = 250.
T = 1/fm

sampleTrials = int(6*fm)

canal1eeg1 = np.asarray(data1[canal6Data1Name])
canal2eeg1 = np.asarray(data1[canal7Data1Name])

trials = 9
trialDuration = 6 #seg

canal1eeg1 = canal1eeg1[:int(fm*trialDuration*trials)].reshape(int(fm*trialDuration),trials)
canal2eeg1 = canal2eeg1[:int(fm*trialDuration*trials)].reshape(int(fm*trialDuration),trials)

t = np.arange(0, int(trialDuration*fm))*T
plt.plot(t,canal1eeg1[:,0])
plt.show()

blancos = 1
canales = 2
samples = canal1eeg1.shape[0]
eeg1 = np.zeros((blancos,canales,samples,trials))
eeg1[:,0,:,:] = canal1eeg1
eeg1[:,1,:,:] = canal2eeg1

eeg1Filtrado = filterEEG(eeg1, lfrec = 7., hfrec = 18., orden = 4, fm  = 250.0)

plt.plot(t,eeg1Filtrado[0,0,:,0])
plt.show()

"""TEST"""

canal1eeg1 = np.asarray(data2[canal7Data1Name])[int(4*fm):]
canal2eeg1 = np.asarray(data2[canal8Data1Name])[int(4*fm):]

canal1eeg1test = []
canal2eeg1test = []
shift = 0
for trial in range(trials):
    canal1eeg1test.append(canal1eeg1[shift:int(fm*4)+shift])
    canal2eeg1test.append(canal2eeg1[shift:int(fm*4)+shift])
    shift += int(fm*trialDuration)
    
canal1eeg1test = np.asarray(canal1eeg1test).T
canal2eeg1test = np.asarray(canal2eeg1test).T



t = np.arange(0, int(4*fm))*T
plt.plot(t,canal1eeg1test[:,0])
plt.show()

blancos = 1
canales = 2
samples = canal1eeg1test.shape[0]
eeg1 = np.zeros((blancos,canales,samples,trials))
eeg1[:,0,:,:] = canal1eeg1test
eeg1[:,1,:,:] = canal2eeg1test

eeg1Filtrado = filterEEG(eeg1, lfrec = 7., hfrec = 18., orden = 4, fm  = 250.0)

plt.plot(t,eeg1Filtrado[0,0,:,0])
plt.show()

resolution = fm/eeg1.shape[2]

PRE_PROCES_PARAMS = {
                'lfrec': 7.,
                'hfrec': 18.,
                'order': 4,
                'sampling_rate': fm,
                'window': 6,
                'shiftLen':6
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 7.0,
                'end_frequency': 18.0,
                'sampling_rate': fm
                }

fftpar = {
    'resolución': resolution,
    'frecuencia inicio': 7.,
    'frecuencia final': 18.0,
    'fm': fm
} #parámetros importantes para aplicar la FFT

frecStimulus = np.array([14])

ventana = 6
solapamiento = ventana*1
canal = 1

#Realizo la segmentación de mi señal de EEG con ventana y el solapamiento dados
eeg1Segmented = segmentingEEG(eeg1, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

espectroEEG1 = computeMagnitudSpectrum(eeg1Segmented, FFT_PARAMS)

#Grafico el espectro para todos los blancos para el canal propuesto
# plotSpectrum(espectroEEG1, resolution, 12, 1, 1, frecStimulus,
#               startFrecGraph = FFT_PARAMS['start_frequency'],
#               save = False, title = "", folder = "figs")

plotOneSpectrum(espectroEEG1, resolution, 1, 1, 0, [14],
                startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro de los canales 8 a 16 -filtrados- de la Synthetic Board",
              folder = "figs")