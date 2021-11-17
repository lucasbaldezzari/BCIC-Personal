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
        nsamples = eeg.shape[0]
        ntrials = eeg.shape[1]
        signalFilteredbyBank = np.zeros((nclases,nsamples,ntrials))

        for clase, frecuencia in enumerate(frecStimulus):   
                low = (frecuencia-bw/2)/nyquist
                high = (frecuencia+bw/2)/nyquist
                b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
                signalFilteredbyBank[clase] = filtfilt(b, a, eeg, axis = axis) #filtramos

        return signalFilteredbyBank

def computWelchPSD(signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):

        anchoVentana = int(fm*anchoVentana) #fm * segundos
        ventana = ventana(anchoVentana)

        signalSampleFrec, signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='mean',axis = axis, scaling = "density")

        return signalSampleFrec, signalPSD

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG\WM\ses1")

trials = 15
fm = 200.
duration = 5 #sec
samplePoints = int(fm*duration)
channels = 4

subjects = [1]
filenames = ["S3_R1_S2_E6", "S3-R1-S1-E7"]
allData = fa.loadData(path = path, filenames = filenames)

name = "S3_R1_S2_E6" #nombre de los datos a analizar}
stimuli = [7,9] #lista de estímulos
estim = [7] #L7e pasamos un estímulo para que grafique una linea vertical

eeg = allData[name]['eeg'][:,:1,:,:]

#Chequeamos información del registro eeg 1
print(allData[name]["generalInformation"])
print(f"Forma de los datos {eeg.shape}")

#Filtramos la señal de eeg para eeg 1
eeg = eeg - eeg.mean(axis = 2, keepdims=True)


resolution = np.round(fm/eeg.shape[2], 4)

PRE_PROCES_PARAMS = {
                'lfrec': 4.,
                'hfrec': 20.,
                'order': 6,
                'sampling_rate': fm,
                'window': duration,
                'shiftLen':duration
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.,
                'end_frequency': 20.0,
                'sampling_rate': fm
                }

window = 5 #sec
ti = 0.5 #en segundos
tf = 0.5 #en segundos
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

eegVentaneados = {'eeg1':eeg, 'eeg2': eeg, 'eeg3': eeg}

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
trial = 5
for i, eegVentaneado in enumerate(eegVentaneados):
        print(eegVentaneado)
        plots[i].plot(t, eegVentaneados[eegVentaneado][0,0,:,trial-1], label = listaVentanas[i], color = "#403e7d")
        plots[i].set_ylabel('Amplitud [uV]')
        plots[i].set_xlabel('tiempo [seg]')
        plots[i].xaxis.grid(True)
        plots[i].legend()
plt.show()

# #eeg data segmentation
# eegSegmented = segmentingEEG(eegFiltered, PRE_PROCES_PARAMS["window"],
#                              PRE_PROCES_PARAMS["shiftLen"],
#                              PRE_PROCES_PARAMS["sampling_rate"])

# MSF1 = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
# C = computeComplexSpectrum(eegSegmented, FFT_PARAMS)

# fft_axis = np.arange(MSF1.shape[0]) * resolution
# plt.plot(fft_axis, MSF1[:,0,0,:5,0])
# plt.show()


########################################################################
#Graficamos espectro para los cuatro canales para un trial en particular
########################################################################
## TODO
## Quedarme con los dos primeros canales. Promediar sobre los dos primeros canales. Aplicar banco de filtros. Aplicar Welch.

ventana = windows.hamming
anchoVentana = 1
frecStimulus = np.array([8,9])
nclases = len(frecStimulus)
nsamples = int(duration*fm)

eegFiltered = eegFiltered.reshape(eegFiltered.shape[1],eegFiltered.shape[2], eegFiltered.shape[3])
avgeeg = eegFiltered.mean(axis = 0)

trial = 5

databanked = applyFilterBank(avgeeg, frecStimulus, bw = 2, order = 4, axis = 0)

# plt.plot(databanked[1])
# plt.show()

signalSampleFrec, signalPSD = computWelchPSD(databanked, fm, ventana, anchoVentana, average = "median", axis = 1)

plt.plot(signalSampleFrec, signalPSD.mean(axis = 2)[0])
plt.plot(signalSampleFrec, signalPSD.mean(axis = 2)[1])
# plt.plot(signalSampleFrec, signalPSD.mean(axis = 2)[2])
plt.show()

# plt.plot(signalSampleFrec, signalPSD.mean(axis = 2).swapaxes(0,1))
# plt.show()

# ########################################################################
# #Graficamos espectro para los cuatro canales promediando los trials
# ########################################################################

# canales = [1,2,3,4]

# title = f"Espectro - Trials promediados - sujeto {name}"
# fig, plots = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw=dict(hspace=0.45, wspace=0.3))
# plots = plots.reshape(-1)
# fig.suptitle(title, fontsize = 16)

# for canal in range(len(canales)):
#         fft_axis = np.arange(MSF1.shape[0]) * resolution
#         # plots[canal].plot(fft_axis + FFT_PARAMS["start_frequency"],
#         #                         np.mean(np.squeeze(MSF1[:, canal, :, :, :]),
#         #                                 axis=1), color = "#403e7d")
#         plots[canal].plot(fft_axis + FFT_PARAMS["start_frequency"],
#                                 np.mean(MSF1, axis = 3).reshape(MSF1.shape[0], MSF1.shape[1])[:,canal]
#                                 , color = "#403e7d")
#         plots[canal].set_xlabel('Frecuencia [Hz]')
#         plots[canal].set_ylabel('Amplitud [uV]')
#         plots[canal].set_title(f'Estímulo {estim[0]} Hz del sujeto canal {canal + 1}')
#         plots[canal].xaxis.grid(True)
#         plots[canal].axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
#                                 label = "Frec. Estímulo",
#                                 linestyle='--', color = "#e37165", alpha = 0.9)
#         plots[canal].legend()

# plt.show()

# ########################################################################
# #Graficamos espectro para los cuatro canales para un trial en particular
# ########################################################################

# canales = [1,2,3,4]
# trial = 3

# title = f"Espectro - Trial número {trial} - sujeto {name}"
# fig, plots = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw=dict(hspace=0.45, wspace=0.3))
# plots = plots.reshape(-1)
# fig.suptitle(title, fontsize = 16)

# for canal in range(len(canales)):
#         fft_axis = np.arange(MSF1.shape[0]) * resolution
#         plots[canal].plot(fft_axis + FFT_PARAMS["start_frequency"],
#                                 MSF1[:, canal, 0, trial - 1, 0] , color = "#403e7d")
#         plots[canal].set_xlabel('Frecuencia [Hz]')
#         plots[canal].set_ylabel('Amplitud [uV]')
#         plots[canal].set_title(f'Estímulo {estim[0]} Hz del sujeto canal {canal + 1}')
#         plots[canal].xaxis.grid(True)
#         plots[canal].axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
#                                 label = "Frec. Estímulo",
#                                 linestyle='--', color = "#e37165", alpha = 0.9)
#         plots[canal].legend()

# plt.show()


# ########################################################################
# #Graficamos espectro canales promediados y un trial
# ########################################################################

# trial = 5

# title = f"Espectro canales promediados - Trial número {trial} - sujeto {name}"
# plt.title(title)
# plt.plot(fft_axis + FFT_PARAMS["start_frequency"],
#                                 MSF1.mean(axis = 1)[:,0,trial-1,0], color = "#403e7d")
# plt.ylabel('Amplitud [uV]')
# plt.axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
#                         label = f"Frec. Estímulo {estim[0]}Hz",
#                         linestyle='--', color = "#e37165", alpha = 0.9)
# plt.legend()
# plt.show()

# ########################################################################
# #graficamos espectro para todos los trials y un canal
# ########################################################################

# canal = 2 #elegimos un canal

# title = f"Espectro para cada trial - Canal {canal} - Estímulo {estim[0]}Hz - Sujeto {name}"

# filas = 3 #Tenemos 15 trials y dividimos el gráfico en 3 filas y 5 columnas
# columnas = 5

# fig, plots = plt.subplots(filas, columnas, figsize=(16, 14), gridspec_kw=dict(hspace=0.35, wspace=0.2))
# plots = plots.reshape(-1)
# fig.suptitle(title, fontsize = 14)

# for trial in range(MSF1.shape[3]):
#         fft_axis = np.arange(MSF1.shape[0]) * resolution
#         plots[trial].plot(fft_axis + FFT_PARAMS["start_frequency"],
#                                 MSF1[:, canal-1, 0, trial, 0] , color = "#403e7d")
#         plots[trial].set_xlabel('Frecuencia [Hz]')
#         plots[trial].set_ylabel('Amplitud [uV]')
#         # plots[trial].set_title(f'Estímulo {estim[0]} Hz del sujeto canal {canal}')
#         plots[trial].xaxis.grid(True)
#         plots[trial].axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
#                                 linestyle='--', color = "#e37165", alpha = 0.9)

# plt.show()