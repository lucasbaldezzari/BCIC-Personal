
import os
import numpy as np
import numpy.matlib as npm
import json

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import pickle

import matplotlib.pyplot as plt

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from utils import norm_mean_std
import fileAdmin as fa

from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

from scipy.signal import butter, filtfilt, iirnotch


actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG\LucasB\ses1")

frecStimulus = np.array([7, 9, 11, 13])

trials = 15
fm = 200.
window = 5 #sec
samplePoints = int(fm*window)
channels = 4

filesRun1 = ["lb-R1-S1-E7","lb-R1-S1-E9", "lb-R1-S1-E11","lb-R1-S1-E13"]
run1 = fa.loadData(path = path, filenames = filesRun1)
filesRun2 = ["lb-R2-S1-E7","lb-R2-S1-E9", "lb-R2-S1-E11","lb-R2-S1-E13"]
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

trainSet = np.concatenate((run1JoinedData[:,:,:,:12], run2JoinedData[:,:,:,:12]), axis = 3)
trainSet = trainSet[:,:2,:,:] #nos quedamos con los primeros dos canales. Forma datos [clases, canales, samples, trials]

nclases = trainSet.shape[0]
nsamples = trainSet.shape[2]
ntrials = trainSet.shape[3]

#Pre procesamiento

trainSetAvgd = np.mean(trainSet, axis = 1) #promediamos sobre los canales. Forma datos ahora [clases, samples, trials]

#Tenemos tantos bancos como frecuencias centrales
order = 4
nyquist = 0.5 * fm

b, a = iirnotch(50., 30, fm) #obtengo los parámetros del filtro
signalFiltered = filtfilt(b, a, trainSetAvgd, axis = 1) #filtramos. Forma datos ahora [clases, samples, trials]

low = 5./nyquist
high = 30./nyquist
b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
signalFiltered = filtfilt(b, a, signalFiltered, axis = 1) #filtramos

#aplicamos filtrado
banco = {}
bw = 2.0 #bandwith

signalFilteredbyBank = np.zeros((nclases,nsamples,ntrials))
for clase, frecuencia in enumerate(frecStimulus):    
    low = (frecuencia-bw/2)/nyquist
    high = (frecuencia+bw/2)/nyquist
    b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
    signalFilteredbyBank[clase] = filtfilt(b, a, signalFiltered[clase], axis = 0) #filtramos

#Computamos el psd de los datos filtrados por el banco.
from scipy.fft import fft

startIndex = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
endIndex = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution'])) + 1
NFFT = int(fm/resolution)
eegPSD = np.abs(np.fft.fft(signalFilteredbyBank, NFFT, axis = 1)[:,startIndex:endIndex,:])**2

plt.plot(eegPSD[1,:,:])
plt.show()


#Usando brainflow sería
##DataFilter.perform_bandpass(trainSetAvgdNotch[0,:,0].ravel(order="C"), 200, 6, bw, 4,FilterTypes.BESSEL.value, 0)

##### Aplicamos la Short Time Fourir transform #####
from scipy.signal import stft, windows

anchoVentana = int(fm*5) #fm * segundos
overlap = int(fm*1) #fm * segundo

ventana = windows.hamming(anchoVentana)

sampleFrec, segments, signalTransformed = stft(signalFiltered, fs=fm,window = ventana, nperseg = anchoVentana, noverlap = overlap, axis = 1) #window = ventana, nperseg = ventana, noverlap = overlap, 

plt.pcolormesh(segments, sampleFrec[:200], np.abs(signalTransformed[2,:200,10,:]), vmin=0, vmax=0.5, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.plot(signalTransformed[0,:,0,:])
plt.show()

##### Computamos la transformada de Welch #####
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

from scipy.signal import welch

anchoVentana = int(fm*5) #fm * segundos
overlap = int(fm*2.5) #fm * segundo

ventana = windows.hamming(anchoVentana)

sampleFrec, signalPSD = welch(signalFiltered, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = 1)

plt.semilogy(sampleFrec, signalPSD[2,:,10])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

sampleFrec, signalPSD = welch(signalFilteredbyBank, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = 1)

trial = 20

plt.plot(sampleFrec, signalPSD[0,:,trial-1], label = f"clase {frecStimulus[0]}")
plt.plot(sampleFrec, signalPSD[1,:,trial-1], label = f"clase {frecStimulus[1]}")
plt.plot(sampleFrec, signalPSD[2,:,trial-1], label = f"clase {frecStimulus[2]}")
plt.plot(sampleFrec, signalPSD[3,:,trial-1], label = f"clase {frecStimulus[3]}")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend()
plt.show()

plt.plot(sampleFrec, signalPSD.mean(axis = 2)[0], label = f"clase {frecStimulus[0]}")
plt.plot(sampleFrec, signalPSD.mean(axis = 2)[1], label = f"clase {frecStimulus[1]}")
plt.plot(sampleFrec, signalPSD.mean(axis = 2)[2], label = f"clase {frecStimulus[2]}")
plt.plot(sampleFrec, signalPSD.mean(axis = 2)[3], label = f"clase {frecStimulus[3]}")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend()
plt.show()

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum
from SVMTrainingModule import SVMTrainingModule