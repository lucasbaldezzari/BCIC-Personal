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
filenames = ["LucasB_7Hz_14Hz_4"]
allData = fa.loadData(path = path, filenames = filenames)

name = "LucasB_7Hz_14Hz_4" #nombre de los datos a analizar}
stimuli = [7,12,15] #lista de estímulos
estim = [9] #L7e pasamos un estímulo para que grafique una linea vertical

frecStimulus = np.array([7,8,10])

eeg = allData[name]['eeg'][:,:,:,:].mean(axis = 1)
eeg = eeg.reshape(1,1,eeg.shape[1],eeg.shape[2])


#Chequeamos información del registro eeg 1
print(allData[name]["generalInformation"])
print(f"Forma de los datos {eeg.shape}")

#Filtramos la señal de eeg para eeg 1
eeg = eeg - eeg.mean(axis = 2, keepdims=True)


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

eeg = eeg[:,:, descarteInicial:descarteFinal, :]

anchoVentana = int((window - ti - tf)*fm) #fm * segundos

ventana1 = windows.hamming(anchoVentana, sym= True)
ventana2 = windows.chebwin(anchoVentana, at = 60, sym= True)
ventana3 = windows.blackman(anchoVentana, sym= True)


ventanas = {
                'ventana1': ventana1,
                'ventana2': ventana2,
                'ventana3': ventana3
                }

### Graficamos ventanas
# title = "Señales de EEG ventaneadas"
# listaVentanas = ["Hamming", "Chebwin", "blackman"]
# fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
# fig.suptitle(title, fontsize = 12)
# t = np.arange(0,anchoVentana/fm,1/fm)
# for i, ventana in enumerate(ventanas):
#         plots[i].plot(ventanas[ventana], label = listaVentanas[i], color = "#403e7d")
#         plots[i].set_ylabel('Amplitud [uV]')
#         plots[i].set_xlabel('tiempo [seg]')
#         plots[i].xaxis.grid(True)
#         plots[i].legend()
# plt.show()

eegVentaneados = {'eeg1':eeg.copy(), 'eeg2': eeg.copy(), 'eeg3': eeg.copy()}

nclases = eeg.shape[0]
nchannels = eeg.shape[1]
ntrials = eeg.shape[3]

for clase in range(nclases):
        for canal in range(nchannels):
                for trial in range(ntrials):
                        eegVentaneados['eeg1'][clase, canal, :, trial] = eeg[clase, canal, :, trial]*ventanas['ventana1']
                        eegVentaneados['eeg2'][clase, canal, :, trial] = eeg[clase, canal, :, trial]*ventanas['ventana2']
                        eegVentaneados['eeg3'][clase, canal, :, trial] = eeg[clase, canal, :, trial]*ventanas['ventana3']

# eegFiltered = filterEEG(eeg, PRE_PROCES_PARAMS["lfrec"],
#                         PRE_PROCES_PARAMS["hfrec"],
#                         PRE_PROCES_PARAMS["order"],
#                         PRE_PROCES_PARAMS["sampling_rate"])

# plt.plot(eegFiltered[0,0,:,0])
# plt.show()

#Filtramos EEGs ventaneados
for eegVentaneado in eegVentaneados:
        eegVentaneados[eegVentaneado] = filterEEG(eegVentaneados[eegVentaneado], PRE_PROCES_PARAMS["lfrec"],
                                        PRE_PROCES_PARAMS["hfrec"],
                                        PRE_PROCES_PARAMS["order"],
                                        PRE_PROCES_PARAMS["sampling_rate"])

title = "Señales de EEG ventaneadas"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 12)
t = np.arange(0,anchoVentana/fm,1/fm)
trial = 2
for i, eegVentaneado in enumerate(eegVentaneados):
        print(eegVentaneado)
        plots[i].plot(t, eegVentaneados[eegVentaneado][0,0,:,trial-1], label = listaVentanas[i], color = "#403e7d")
        plots[i].set_ylabel('Amplitud [uV]')
        plots[i].set_xlabel('tiempo [seg]')
        plots[i].xaxis.grid(True)
        plots[i].legend()
plt.show()

#Segmentamos EEGs ventaneados
eegSegmented = {}

for eegVentaneado in eegVentaneados:
        eegSegmented[eegVentaneado] = segmentingEEG(eegVentaneados[eegVentaneado], PRE_PROCES_PARAMS["window"],
                                        PRE_PROCES_PARAMS["shiftLen"],
                                        PRE_PROCES_PARAMS["sampling_rate"])
MSFs = {}

for eegVentaneado in eegVentaneados:
        MSFs[eegVentaneado] = computeMagnitudSpectrum(eegSegmented[eegVentaneado], FFT_PARAMS)

# #eeg data segmentation
# eegSegmented = segmentingEEG(eegFiltered, PRE_PROCES_PARAMS["window"],
#                              PRE_PROCES_PARAMS["shiftLen"],
#                              PRE_PROCES_PARAMS["sampling_rate"])

# MSF1 = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
# C = computeComplexSpectrum(eegSegmented, FFT_PARAMS)

# fft_axis = np.arange(MSF1.shape[0]) * resolution
# plt.plot(fft_axis, MSF1[:,0,0,:5,0])
# plt.show()

title = "Espectro Fourier de señales de EEG ventaneadas"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 12)
fft_axis = np.arange(MSFs['eeg1'].shape[0]) * resolution
trial = 2
for i, eegVentaneado in enumerate(eegVentaneados):
        print(eegVentaneado)
        plots[i].plot(fft_axis, MSFs[eegVentaneado][:,0,0,trial-1,0], label = listaVentanas[i], color = "#403e7d")
        plots[i].set_ylabel('Amplitud [uV]')
        plots[i].set_xlabel('Frecuencia [Hz]')
        plots[i].xaxis.grid(True)
        plots[i].legend()
plt.show()


########################################################################
###                    Aplicamos Welch
########################################################################

ventana1 = windows.hamming(anchoVentana, sym= True)
ventana2 = windows.chebwin(anchoVentana, at = 60, sym= True)
ventana3 = windows.blackman(anchoVentana, sym= True)

signalSampleFrec1, signalPSD1 = computWelchPSD(eegVentaneados['eeg1'], fm, ventana1, anchoVentana, average = "median", axis = 2)
signalSampleFrec2, signalPSD2 = computWelchPSD(eegVentaneados['eeg2'], fm, ventana2, anchoVentana, average = "median", axis = 2)
signalSampleFrec3, signalPSD3 = computWelchPSD(eegVentaneados['eeg3'], fm, ventana3, anchoVentana, average = "median", axis = 2)

samplesFrec = [signalSampleFrec1, signalSampleFrec2, signalSampleFrec3]
signalPSDs = [signalPSD1, signalPSD2, signalPSD3]

title = "Espectro Fourier de señales de EEG ventaneadas"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 12)
fft_axis = np.arange(MSFs['eeg1'].shape[0]) * resolution
trial = 1
for i in range(len(signalPSDs)):
        plots[i].plot(samplesFrec[i][:120], signalPSDs[i][0,0,:,trial-1][:120], label = listaVentanas[i], color = "#403e7d")
        plots[i].set_ylabel('Amplitud [uV^2/Hz]')
        plots[i].set_xlabel('Frecuencia [Hz]')
        plots[i].xaxis.grid(True)
        plots[i].legend()
plt.show()


########################################################################
###             Aplicamos banco de filtros
########################################################################

# frecStimulus = np.array([8,10,14])

dataBanked = {}
for eegVentaneado in eegVentaneados:
        signal = eegVentaneados[eegVentaneado][0,:,:,:].copy()
        dataBanked[eegVentaneado] = applyFilterBank(signal, frecStimulus, bw = 2, order = 4, axis = 1)

title = "Señales de EEG ventaneadas y filtradas con el banco de filtros"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(len(ventanas), len(frecStimulus), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 12)
t = np.arange(0,anchoVentana/fm,1/fm)
trial = 1
for i, eeg in enumerate(dataBanked):
        plots[i][0].plot(t, dataBanked[eeg][0,:,trial-1], label = f'fc{frecStimulus[0]}')
        plots[i][1].plot(t, dataBanked[eeg][1,:,trial-1], label = f'fc{frecStimulus[1]}')
        plots[i][2].plot(t, dataBanked[eeg][2,:,trial-1], label = f'fc{frecStimulus[2]}')
        plots[i][0].set_ylabel('Amplitud [uV]')
        plots[i][1].set_ylabel('Amplitud [uV]')
        plots[i][2].set_ylabel('Amplitud [uV]')
        plots[i][0].set_xlabel('tiempo [seg]')
        plots[i][1].set_xlabel('tiempo [seg]')
        plots[i][2].set_xlabel('tiempo [seg]')
        plots[i][0].legend()
        plots[i][1].legend()
        plots[i][2].legend()
plt.show()

########################################################################
###        Aplicamos Welch a las señales filtradas con el banco
########################################################################

signalSampleFrec1, signalPSD1 = computWelchPSD(dataBanked['eeg1'], fm, ventana1, anchoVentana, average = "median", axis = 1)
signalSampleFrec2, signalPSD2 = computWelchPSD(dataBanked['eeg2'], fm, ventana2, anchoVentana, average = "median", axis = 1)
signalSampleFrec3, signalPSD3 = computWelchPSD(dataBanked['eeg3'], fm, ventana3, anchoVentana, average = "median", axis = 1)

samplesFrec = [signalSampleFrec1, signalSampleFrec2, signalSampleFrec3]
signalPSDs = [signalPSD1, signalPSD2, signalPSD3]

title = "Espectro con método Welch y señales banqueadas"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(1, len(frecStimulus), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 16)
t = np.arange(0,anchoVentana/fm,1/fm)
trial = 1
for i, signalPSD in enumerate(signalPSDs):
        plots[i].plot(signalSampleFrec1[:120], signalPSD[0,:120,trial-1], label = f'fc{frecStimulus[0]}')
        plots[i].plot(signalSampleFrec2[:120], signalPSD[1,:120,trial-1], label = f'fc{frecStimulus[1]}')
        plots[i].plot(signalSampleFrec3[:120], signalPSD[2,:120,trial-1], label = f'fc{frecStimulus[2]}')
        plots[i].set_title(f'Potencia para {listaVentanas[i]}')
        plots[i].set_ylabel('Amplitud [uV^2/Hz]')
        plots[i].set_xlabel('Frecuencia [Hz]')
        plots[i].legend()
plt.show()

#### Espectros para todos los trials
title = f"Espectro con método Welch y señales banqueadas en frecuencias centrales {frecStimulus}Hz - Estímulo {estim}Hz"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(trials, len(frecStimulus), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 10)
for trial in range(trials):
        for i, signalPSD in enumerate(signalPSDs):
                plots[trial][i].plot(signalSampleFrec1[:80], signalPSD[0,:80,trial], label = f'fc{frecStimulus[0]}')
                plots[trial][i].plot(signalSampleFrec2[:80], signalPSD[1,:80,trial], label = f'fc{frecStimulus[1]}')
                plots[trial][i].plot(signalSampleFrec3[:80], signalPSD[2,:80,trial], label = f'fc{frecStimulus[2]}')
plt.show()

#Aplicando banco de filtros para buscar harmónicos

armonic = frecStimulus*2

dataBanked2 = {}
for eegVentaneado in eegVentaneados:
        signal = eegVentaneados[eegVentaneado][0,:,:,:].copy()
        dataBanked2[eegVentaneado] = applyFilterBank(signal, armonic, bw = 2, order = 4, axis = 1)

title = "Señales de EEG ventaneadas y filtradas con el banco de filtros en los armónicos"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(len(ventanas), len(armonic), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 12)
t = np.arange(0,anchoVentana/fm,1/fm)
trial = 1
for i, eeg in enumerate(dataBanked):
        plots[i][0].plot(t, dataBanked2[eeg][0,:,trial-1], label = f'fc{armonic[0]}')
        plots[i][1].plot(t, dataBanked2[eeg][1,:,trial-1], label = f'fc{armonic[1]}')
        plots[i][2].plot(t, dataBanked2[eeg][2,:,trial-1], label = f'fc{armonic[2]}')
        plots[i][0].set_ylabel('Amplitud [uV]')
        plots[i][1].set_ylabel('Amplitud [uV]')
        plots[i][2].set_ylabel('Amplitud [uV]')
        plots[i][0].set_xlabel('tiempo [seg]')
        plots[i][1].set_xlabel('tiempo [seg]')
        plots[i][2].set_xlabel('tiempo [seg]')
        plots[i][0].legend()
        plots[i][1].legend()
        plots[i][2].legend()
plt.show()

########################################################################
###        Aplicamos Welch a las señales filtradas con el banco
########################################################################

signalSampleFrec1, signalPSD1armonic = computWelchPSD(dataBanked2['eeg1'], fm, ventana1, anchoVentana, average = "median", axis = 1)
signalSampleFrec2, signalPSD2armonic = computWelchPSD(dataBanked2['eeg2'], fm, ventana2, anchoVentana, average = "median", axis = 1)
signalSampleFrec3, signalPSD3armonic = computWelchPSD(dataBanked2['eeg3'], fm, ventana3, anchoVentana, average = "median", axis = 1)

samplesFrec = [signalSampleFrec1, signalSampleFrec2, signalSampleFrec3]
armonicSignalPSDs = [signalPSD1armonic, signalPSD2armonic, signalPSD3armonic]

trial = 1
title = f"Espectro con método Welch y señales banqueadas para trial {trial}"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(1, len(armonic), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 16)
t = np.arange(0,anchoVentana/fm,1/fm)
for i, signalPSD in enumerate(signalPSDs):
        plots[i].plot(signalSampleFrec1[:150], signalPSD1armonic[0,:150,trial-1], label = f'fc{armonic[0]}')
        plots[i].plot(signalSampleFrec2[:150], signalPSD2armonic[1,:150,trial-1], label = f'fc{armonic[1]}')
        plots[i].plot(signalSampleFrec3[:150], signalPSD3armonic[2,:150,trial-1], label = f'fc{armonic[2]}')
        plots[i].set_title(f'Potencia para {listaVentanas[i]}')
        plots[i].set_ylabel('Amplitud [uV^2/Hz]')
        plots[i].set_xlabel('Frecuencia [Hz]')
        plots[i].legend()
plt.show()

#### Espectros para todos los trials
title = f"Espectro con método Welch y señales banqueadas en frecuencias centrales {frecStimulus}Hz - Estímulo {estim}Hz"
listaVentanas = ["Hamming", "Chebwin", "blackman"]
fig, plots = plt.subplots(trials, len(frecStimulus), gridspec_kw=dict(hspace=0.45, wspace=0.3))
fig.suptitle(title, fontsize = 10)
for trial in range(trials):
        for i, signalPSD in enumerate(signalPSDs):
                plots[trial][i].plot(signalSampleFrec1[:150], signalPSD3armonic[0,:150,trial], label = f'fc{frecStimulus[0]}')
                plots[trial][i].plot(signalSampleFrec2[:150], signalPSD3armonic[1,:150,trial], label = f'fc{frecStimulus[1]}')
                plots[trial][i].plot(signalSampleFrec3[:150], signalPSD3armonic[2,:150,trial], label = f'fc{frecStimulus[2]}')
plt.show()

########################################################################
###        Fusionando espectros
########################################################################

# completeSpectrum1 = np.sum((signalPSD1, signalPSD1armonic), axis = 1)
# test = np.array((signalPSD1, signalPSD1armonic))

# spectrumFor12 = []

# for signalPSD in signalPSDs:
#         aux = np.array((signalPSD1, signalPSD1armonic))
#         spectrumFor12.append(np.sum(aux, axis = 0))

# trial = 3
# title = f"Espectros con frecuencia central y primer armónico para trial {trial}"
# listaVentanas = ["Hamming", "Chebwin", "blackman"]
# fig, plots = plt.subplots(1, len(spectrumFor12), figsize=(12, 6), gridspec_kw=dict(hspace=0.45, wspace=0.3))
# fig.suptitle(title, fontsize = 16)
# t = np.arange(0,anchoVentana/fm,1/fm)
# for i, signalPSD in enumerate(signalPSDs):
#         plots[i].plot(signalSampleFrec1[:120], spectrumFor12[0][0,:120,trial-1], label = f'fc{frecStimulus[0]}-fc{armonic[0]}')
#         plots[i].plot(signalSampleFrec2[:120], spectrumFor12[1][1,:120,trial-1], label = f'fc{frecStimulus[1]}-fc{armonic[0]}')
#         plots[i].plot(signalSampleFrec3[:120], spectrumFor12[2][2,:120,trial-1], label = f'fc{frecStimulus[2]}-fc{armonic[0]}')
#         plots[i].set_title(f'Potencia para {listaVentanas[i]}')
#         plots[i].set_ylabel('Amplitud [uV^2/Hz]')
#         plots[i].set_xlabel('Frecuencia [Hz]')
#         plots[i].legend()
# plt.show()

# #### Espectros para todos los trials
# title = f"Espectros con frecuencia central y primer armónico para trial {trial} - Estímulo {estim}Hz"
# listaVentanas = ["Hamming", "Chebwin", "blackman"]
# fig, plots = plt.subplots(trials, len(frecStimulus), gridspec_kw=dict(hspace=0.45, wspace=0.3))
# fig.suptitle(title, fontsize = 10)
# for trial in range(trials):
#         for i, signalPSD in enumerate(signalPSDs):
#                 plots[trial][i].plot(signalSampleFrec1[:80], spectrumFor12[0][0,:80,trial], label = f'fc{frecStimulus[0]}')
#                 plots[trial][i].plot(signalSampleFrec2[:80], spectrumFor12[0][1,:80,trial], label = f'fc{frecStimulus[1]}')
#                 plots[trial][i].plot(signalSampleFrec3[:80], spectrumFor12[0][2,:80,trial], label = f'fc{frecStimulus[2]}')
# plt.show()
