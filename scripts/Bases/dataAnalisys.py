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
from utils import norm_mean_std

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

def applyFilterBank(eeg, frecStimulus, bw = 2.0, order = 4, axis = 1):

        nyquist = 0.5 * fm
        nclases = len(frecStimulus)
        nsamples = eeg.shape[1]
        ntrials = eeg.shape[2]
        signalFilteredbyBank = np.zeros((nclases,nsamples,ntrials))

        for clase, frecuencia in enumerate(frecStimulus):
                low = (frecuencia-bw/2)/nyquist
                high = (frecuencia+bw/2)/nyquist
                b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
                signalFilteredbyBank[clase] = filtfilt(b, a, eeg, axis = axis) #filtramos

        return signalFilteredbyBank

def computWelchPSD(signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):

        # anchoVentana = int(fm*anchoVentana) #fm * segundos
        # ventana = ventana(anchoVentana)

        signalSampleFrec, signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='mean',axis = axis, scaling = "density")

        return signalSampleFrec, signalPSD

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG")

trials = 5
fm = 200.
duration = 4 #sec
samplePoints = int(fm*duration)
channels = 2

subjects = [1]
filenames = ["walter_s1_r1_10hz"]
allData = fa.loadData(path = path, filenames = filenames)

name = "walter_s1_r1_10hz" #nombre de los datos a analizar}
stimuli = [7,8.5,10] #lista de estímulos
estim = [8.5] #L7e pasamos un estímulo para que grafique una linea vertical

frecStimulus = np.array([7,8.5,10])

eeg = allData[name]['eeg'][:,:,:,:]
eegO1O2 = eeg.mean(axis = 1)
eegO1O2 = eegO1O2.reshape(1,1,eegO1O2.shape[1],eegO1O2.shape[2])
eegO1 = eeg[:,0,:,:].reshape(1,1,eeg[:,0,:,:].shape[1],eeg[:,0,:,:].shape[2])
eegO2 = eeg[:,1,:,:].reshape(1,1,eeg[:,1,:,:].shape[1],eeg[:,1,:,:].shape[2])

#Chequeamos información del registro eeg 1
print(allData[name]["generalInformation"])
print(f"Forma de los datos {eeg.shape}")

#Filtramos la señal de eeg para eeg 1
eegO1O2 = eegO1O2 - eegO1O2.mean(axis = 2, keepdims=True)
eegO1 = eegO1 - eegO1.mean(axis = 2, keepdims=True)
eegO2 = eegO2 - eegO2.mean(axis = 2, keepdims=True)

subtitles = ["O1 y O2 promediados", "O1", "O2"]

resolution = np.round(fm/eeg.shape[2], 4)

PRE_PROCES_PARAMS = {
                'lfrec': 5.,
                'hfrec': 30.,
                'order': 4,
                'sampling_rate': fm,
                'window': duration,
                'shiftLen':duration
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.,
                'end_frequency': 30.0,
                'sampling_rate': fm
                }

window = duration #sec
ti = 0.3 #en segundos
tf = 0.1 #en segundos
descarteInicial = int(fm*ti) #en segundos
descarteFinal = int(window*fm)-int(tf*fm) #en segundos

anchoVentana = int((window - ti - tf)*fm)

eegO1O2 = eegO1O2[:,:, descarteInicial:descarteFinal, :]
eegO1 = eegO1[:,:, descarteInicial:descarteFinal, :]
eegO2 = eegO2[:,:, descarteInicial:descarteFinal, :]

listaeeg = [eegO1O2, eegO1, eegO2]

#Filtramos EEGs
for i in range(len(listaeeg)):
        listaeeg[i] = filterEEG(listaeeg[i], PRE_PROCES_PARAMS["lfrec"],
                                        PRE_PROCES_PARAMS["hfrec"],
                                        PRE_PROCES_PARAMS["order"],
                                        PRE_PROCES_PARAMS["sampling_rate"])

########################################################################
###              Seleccionamos qué queremos filtrar
########################################################################
plotFilteredEEGs = True
plotFFT = True
plotWelchPSD = True

trial = 2
###

if plotFilteredEEGs == True:
    title = f"Señales EEG por canal - Trial {trial}"
    subtitles = ["O1 y O2 promediados", "O1", "O2"]
    fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    fig.suptitle(title, fontsize = 12)
    t = np.arange(0,anchoVentana/fm,1/fm)
    for i, actualeeg in enumerate(listaeeg):
            plots[i].plot(t, actualeeg[0,0,:,trial-1], label = subtitles[i], color = "#403e7d")
            plots[i].set_ylabel('Amplitud [uV]')
            plots[i].set_xlabel('tiempo [seg]')
            plots[i].xaxis.grid(True)
            plots[i].legend()
    plt.show()

########################################################################
###                    Aplicamos FFT
########################################################################

#Segmentamos EEGs
eegSegmented = dict()
subtitles = ["O1 y O2 promediados", "O1", "O2"]

for i in range(len(listaeeg)):
        eegSegmented[subtitles[i]] = segmentingEEG(listaeeg[i], PRE_PROCES_PARAMS["window"],
                                        PRE_PROCES_PARAMS["shiftLen"],
                                        PRE_PROCES_PARAMS["sampling_rate"])
MSFs = {}

for i in range(len(listaeeg)):
        MSFs[subtitles[i]] = computeMagnitudSpectrum(eegSegmented[subtitles[i]], FFT_PARAMS)

if plotFFT == True:
    title = f"Espectro Fourier de señales de EEG trial - Trial {trial} - Datos sin banquear"
    subtitles = ["O1 y O2 promediados", "O1", "O2"]
    fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    fig.suptitle(title, fontsize = 12)
    fft_axis = np.arange(MSFs["O1"].shape[0]) * resolution
    for i, subtitle in enumerate(subtitles):
            plots[i].plot(fft_axis, MSFs[subtitle][:,0,0,trial-1,0], label = subtitle, color = "#403e7d")
            plots[i].set_ylabel('Amplitud [uV]')
            plots[i].set_xlabel('Frecuencia [Hz]')
            plots[i].xaxis.grid(True)
            plots[i].legend()
    plt.show()

########################################################################
###                    Aplicamos Welch
########################################################################

ventana = windows.blackman(anchoVentana, sym= True)

signalSampleFrec1, O1O2PSD1 = computWelchPSD(listaeeg[0], fm, ventana, anchoVentana, average = "median", axis = 2)
signalSampleFrec2, O1PSD2 = computWelchPSD(listaeeg[1], fm, ventana, anchoVentana, average = "median", axis = 2)
signalSampleFrec3, O2PSD3 = computWelchPSD(listaeeg[2], fm, ventana, anchoVentana, average = "median", axis = 2)

samplesFrec = [signalSampleFrec1, signalSampleFrec2, signalSampleFrec3]
signalPSDs = [O1O2PSD1, O1PSD2, O2PSD3]

if plotWelchPSD == True:
    title = f"PSD para Welch - Trial {trial} - Datos sin banquear"
    subtitles = ["O1 y O2 promediados", "O1", "O2"]
    fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    fig.suptitle(title, fontsize = 12)
    for i in range(len(signalPSDs)):
            plots[i].plot(samplesFrec[i][:120], signalPSDs[i][0,0,:,trial-1][:120], label = subtitles[i], color = "#403e7d")
            plots[i].set_ylabel('Amplitud [uV^2/Hz]')
            plots[i].set_xlabel('Frecuencia [Hz]')
            plots[i].xaxis.grid(True)
            plots[i].legend()
    plt.show()

########################################################################
###             Aplicamos banco de filtros en frecuencia central
########################################################################

dataBanked = {}
for i in range(len(listaeeg)):
        signal = listaeeg[i][0,:,:,:].copy()
        dataBanked[subtitles[i]] = applyFilterBank(signal, frecStimulus, bw = 2, order = 4, axis = 1)

########################################################################
###        Aplicamos Welch a las señales filtradas con el banco
########################################################################

signalSampleFrec1, O1O2PSDcentral = computWelchPSD(dataBanked[subtitles[0]], fm, ventana, anchoVentana, average = "median", axis = 1)
signalSampleFrec2, O1PSDcentral = computWelchPSD(dataBanked[subtitles[1]], fm, ventana, anchoVentana, average = "median", axis = 1)
signalSampleFrec3, O2PSDcentral = computWelchPSD(dataBanked[subtitles[2]], fm, ventana, anchoVentana, average = "median", axis = 1)

samplesFrec = [signalSampleFrec1, signalSampleFrec2, signalSampleFrec3]
signalPSDs = [O1O2PSDcentral, O1PSDcentral, O2PSDcentral]

title = "Espectro con método Welch y señales banqueadas en frecuencia central"
fig, plots = plt.subplots(1, len(frecStimulus), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 16)
for i, signalPSD in enumerate(signalPSDs):
        plots[i].plot(signalSampleFrec1[:120], signalPSD[0,:120,trial-1], label = f'fc{frecStimulus[0]}')
        plots[i].plot(signalSampleFrec2[:120], signalPSD[1,:120,trial-1], label = f'fc{frecStimulus[1]}')
        plots[i].plot(signalSampleFrec3[:120], signalPSD[2,:120,trial-1], label = f'fc{frecStimulus[2]}')
        plots[i].set_title(f'Potencia para {subtitles[i]}')
        plots[i].set_ylabel('Amplitud [uV^2/Hz]')
        plots[i].set_xlabel('Frecuencia [Hz]')
        plots[i].legend()
plt.show()

#### Espectros para todos los trials
title = f"Espectro con método Welch y señales banqueadas en frecuencias centrales {frecStimulus}Hz - Estímulo {estim}Hz"
fig, plots = plt.subplots(trials, len(frecStimulus), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 10)
for trial in range(trials):
        for i, signalPSD in enumerate(signalPSDs):
                plots[trial][i].plot(signalSampleFrec1[:80], signalPSD[0,:80,trial], label = f'fc{frecStimulus[0]}')
                plots[trial][i].plot(signalSampleFrec2[:80], signalPSD[1,:80,trial], label = f'fc{frecStimulus[1]}')
                plots[trial][i].plot(signalSampleFrec3[:80], signalPSD[2,:80,trial], label = f'fc{frecStimulus[2]}')
                plots[trial][i].legend()
plt.show()

########################################################################
###             Aplicamos banco de filtros en 1er armónico
########################################################################

dataBanked2 = {}
armonics = frecStimulus*2

for i in range(len(listaeeg)):
        signal = listaeeg[i][0,:,:,:].copy()
        dataBanked2[subtitles[i]] = applyFilterBank(signal, armonics, bw = 2, order = 4, axis = 1)

signalSampleFrec1, O1O2PSD1armonic = computWelchPSD(dataBanked2[subtitles[0]], fm, ventana, anchoVentana, average = "median", axis = 1)
signalSampleFrec2, O1PSD21armonic = computWelchPSD(dataBanked2[subtitles[1]], fm, ventana, anchoVentana, average = "median", axis = 1)
signalSampleFrec3, O2PSD31armonic = computWelchPSD(dataBanked2[subtitles[2]], fm, ventana, anchoVentana, average = "median", axis = 1)

samplesFrec = [signalSampleFrec1, signalSampleFrec2, signalSampleFrec3]
signalPSDs = [O1O2PSD1armonic, O1PSD21armonic, O2PSD31armonic]

title = "Espectro con método Welch y señales banqueadas en 1er armónico"
fig, plots = plt.subplots(1, len(armonics), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 16)
for i, signalPSD in enumerate(signalPSDs):
        plots[i].plot(signalSampleFrec1[:120], signalPSD[0,:120,trial-1], label = f'fc {armonics[0]}')
        plots[i].plot(signalSampleFrec2[:120], signalPSD[1,:120,trial-1], label = f'fc {armonics[1]}')
        plots[i].plot(signalSampleFrec3[:120], signalPSD[2,:120,trial-1], label = f'fc {armonics[2]}')
        plots[i].set_title(f'Potencia para {subtitles[i]}')
        plots[i].set_ylabel('Amplitud [uV^2/Hz]')
        plots[i].set_xlabel('Frecuencia [Hz]')
        plots[i].legend()
plt.show()

#### Espectros para todos los trials
title = f"Espectro con método Welch y señales banqueadas en frecuencias centrales {frecStimulus}Hz - Estímulo {estim}Hz"
fig, plots = plt.subplots(trials, len(frecStimulus), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 10)
for trial in range(trials):
        for i, signalPSD in enumerate(signalPSDs):
                plots[trial][i].plot(signalSampleFrec1[:80], signalPSD[0,:80,trial], label = f'fc{armonics[0]}')
                plots[trial][i].plot(signalSampleFrec2[:80], signalPSD[1,:80,trial], label = f'fc{armonics[1]}')
                plots[trial][i].plot(signalSampleFrec3[:80], signalPSD[2,:80,trial], label = f'fc{armonics[2]}')
                plots[trial][i].legend()
plt.show()