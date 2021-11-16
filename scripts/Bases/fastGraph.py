
import os
import numpy as np

import fileAdmin as fa
from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum, plotEEG
from utils import norm_mean_std

from scipy.signal import windows

from scipy.signal import welch

import matplotlib.pyplot as plt

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG\FernandoMoreira")

trials = 1
fm = 200.
window = 5 #sec
samplePoints = int(fm*window)
channels = 4
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["S1-R1-S1-E7","S1-R1-S1-E9"]
allData = fa.loadData(path = path, filenames = filenames)

eeg1 = allData["S1-R1-S1-E7"]["eeg"]
eeg2 = allData["S1-R1-S1-E9"]["eeg"]

eeg1 = eeg1[:,:1,:,:]
eeg2 = eeg2[:,:1,:,:]   

# eeg1 = eeg1[:,:, descarteInicial:descarteFinal, :]
# eeg2 = eeg2[:,:, descarteInicial:descarteFinal, :]

plotEEG(eeg1, sujeto = 1, trial = 1, blanco = 1,
            fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

# Restamos la media de la señal
eeg1 = eeg1 - eeg1.mean(axis = 2, keepdims=True)
eeg2 = eeg2 - eeg2.mean(axis = 2, keepdims=True)

plotEEG(eeg1, sujeto = 1, trial = 1, blanco = 1,
            fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

anchoVentana = int(fm*window)
ventana = windows.hamming(anchoVentana, sym= True)

ventana = windows.chebwin(anchoVentana, at = 60, sym= True)

ventana = windows.blackman(anchoVentana, sym= True)

nclases = eeg1.shape[0]
nchannels = eeg1.shape[1]
ntrials = eeg1.shape[3]

# for clase in range(nclases):
#     for canal in range(nchannels):
#         for trial in range(ntrials):
#             eeg1[clase, canal, :, trial] = eeg1[clase, canal, :, trial]*ventana
#             eeg2[clase, canal, :, trial] = eeg2[clase, canal, :, trial]*ventana

eeg1filtered = filterEEG(eeg1, 4, 30, 6, 50., fm = fm)
eeg2filtered = filterEEG(eeg2, 4, 30, 6, 50., fm = fm)

plotEEG(eeg1filtered, sujeto = 1, trial = 1, blanco = 1,
            fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

anchoVentana = int(fm*window/2)
ventana = windows.hamming(anchoVentana)
ventana = windows.blackman(anchoVentana, sym= True)

signalSampleFrec1, signalPSD1 = welch(eeg1filtered, fs = fm, window = ventana, nperseg = anchoVentana, average='median', axis = 2)
signalSampleFrec2, signalPSD2 = welch(eeg2filtered, fs = fm, window = ventana, nperseg = anchoVentana, average='median', axis = 2)

plt.plot(signalSampleFrec1, signalPSD1.mean(axis = 3)[0,0,:], label = "eeg1")      
plt.plot(signalSampleFrec2, signalPSD2.mean(axis = 3)[0,0,:], label = "eeg2")             
plt.legend()
plt.show()

# ti = 1 #en segundos
# tf = 1 #en segundos
# descarteInicial = int(fm*ti) #en segundos
# descarteFinal = int(window*fm)-int(tf*fm) #en segundos

# eeg1FilteredCut = eeg1[:,:, descarteInicial:descarteFinal, :]
# eeg2FilteredCut = eeg2[:,:, descarteInicial:descarteFinal, :]

# nclases = eeg1.shape[0]
# nchannels = eeg1.shape[1]
# ntrials = eeg1.shape[3]

# plotEEG(eeg1FilteredCut, sujeto = 1, trial = 1, blanco = 1,
#             fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

# eeg1FilteredCut = filterEEG(eeg1FilteredCut, 4, 38, 8, 50., fm = fm)
# eeg1FilteredCut = filterEEG(eeg2FilteredCut, 4, 38, 8, 50., fm = fm)

# plotEEG(eeg1FilteredCut, sujeto = 1, trial = 1, blanco = 1,
#             fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

# signalSampleFrec1, signalPSD1 = welch(eeg1FilteredCut, fs = fm, window = ventana, nperseg = anchoVentana, average='median', axis = 2)
# signalSampleFrec2, signalPSD2 = welch(eeg2FilteredCut, fs = fm, window = ventana, nperseg = anchoVentana, average='median', axis = 2)

# plt.plot(signalSampleFrec1, signalPSD1.mean(axis = 3)[0,0,:])      
# plt.plot(signalSampleFrec2, signalPSD2.mean(axis = 3)[0,0,:])             
# plt.show()
