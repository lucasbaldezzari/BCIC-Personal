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

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG")

trials = 10
fm = 200.
window = 4 #sec
samplePoints = int(fm*window)
channels = 4
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["lucasB_11hz_ultracortex"]
allData = fa.loadData(path = path, filenames = filenames)

name = "lucasB_11hz_ultracortex" #nombre de los datos a analizar}
stimuli = [7,9,11,13] #lista de estímulos
estim = [11] #L7e pasamos un estímulo para que grafique una linea vertical

eeg = allData[name]['eeg'][:,:4,:,:] #utilizamos solo los dos primeros canales
#eeg = norm_mean_std(eeg) #normalizamos los datos

#Chequeamos información del registro eeg 1
print(allData[name]["generalInformation"])
print(f"Forma de los datos {eeg.shape}")

#Filtramos la señal de eeg para eeg 1

resolution = np.round(fm/eeg.shape[2], 4)

PRE_PROCES_PARAMS = {
                'lfrec': 8.,
                'hfrec': 30.,
                'order': 4,
                'sampling_rate': fm,
                'window': window,
                'shiftLen':window
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 0.,
                'end_frequency': 30.0,
                'sampling_rate': fm
                }

eegFiltered = filterEEG(eeg, PRE_PROCES_PARAMS["lfrec"],
                        PRE_PROCES_PARAMS["hfrec"],
                        PRE_PROCES_PARAMS["order"],
                        PRE_PROCES_PARAMS["sampling_rate"])

##Computamos el espectro de frecuencias

#eeg data segmentation
eegSegmented = segmentingEEG(eegFiltered, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

MSF1 = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)

########################################################################
#Graficamos espectro para los cuatro canales promediando los trials
########################################################################

canales = [1,2,3,4]

title = f"Espectro - Trials promediados - sujeto {name}"
fig, plots = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw=dict(hspace=0.45, wspace=0.3))
plots = plots.reshape(-1)
fig.suptitle(title, fontsize = 16)

for canal in range(len(canales)):
        fft_axis = np.arange(MSF1.shape[0]) * resolution
        # plots[canal].plot(fft_axis + FFT_PARAMS["start_frequency"],
        #                         np.mean(np.squeeze(MSF1[:, canal, :, :, :]),
        #                                 axis=1), color = "#403e7d")
        plots[canal].plot(fft_axis + FFT_PARAMS["start_frequency"],
                                np.mean(MSF1, axis = 3).reshape(MSF1.shape[0], MSF1.shape[1])[:,canal]
                                , color = "#403e7d")
        plots[canal].set_xlabel('Frecuencia [Hz]')
        plots[canal].set_ylabel('Amplitud [uV]')
        plots[canal].set_title(f'Estímulo {estim[0]} Hz del sujeto canal {canal + 1}')
        plots[canal].xaxis.grid(True)
        plots[canal].axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
                                label = "Frec. Estímulo",
                                linestyle='--', color = "#e37165", alpha = 0.9)
        plots[canal].legend()

plt.show()

########################################################################
#Graficamos espectro para los cuatro canales para un trial en particular
########################################################################

canales = [1,2,3,4]
trial = 3

title = f"Espectro - Trial número {trial} - sujeto {name}"
fig, plots = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw=dict(hspace=0.45, wspace=0.3))
plots = plots.reshape(-1)
fig.suptitle(title, fontsize = 16)

for canal in range(len(canales)):
        fft_axis = np.arange(MSF1.shape[0]) * resolution
        plots[canal].plot(fft_axis + FFT_PARAMS["start_frequency"],
                                MSF1[:, canal, 0, trial - 1, 0] , color = "#403e7d")
        plots[canal].set_xlabel('Frecuencia [Hz]')
        plots[canal].set_ylabel('Amplitud [uV]')
        plots[canal].set_title(f'Estímulo {estim[0]} Hz del sujeto canal {canal + 1}')
        plots[canal].xaxis.grid(True)
        plots[canal].axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
                                label = "Frec. Estímulo",
                                linestyle='--', color = "#e37165", alpha = 0.9)
        plots[canal].legend()

plt.show()


########################################################################
#Graficamos espectro canales promediados y un trial
########################################################################

trial = 4

title = f"Espectro canales promediados - Trial número {trial} - sujeto {name}"
plt.title(title)
plt.plot(fft_axis + FFT_PARAMS["start_frequency"],
                                MSF1.mean(axis = 1)[:,0,trial-1,0], color = "#403e7d")
plt.ylabel('Amplitud [uV]')
plt.axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
                        label = f"Frec. Estímulo {estim[0]}Hz",
                        linestyle='--', color = "#e37165", alpha = 0.9)
plt.legend()
plt.show()

########################################################################
#graficamos espectro para todos los trials y un canal
########################################################################

canal = 1 #elegimos un canal

title = f"Espectro para cada trial - Canal {canal} - Estímulo {estim[0]}Hz - Sujeto {name}"

filas = 3 #Tenemos 15 trials y dividimos el gráfico en 3 filas y 5 columnas
columnas = 5

fig, plots = plt.subplots(filas, columnas, figsize=(16, 14), gridspec_kw=dict(hspace=0.35, wspace=0.2))
plots = plots.reshape(-1)
fig.suptitle(title, fontsize = 14)

for trial in range(MSF1.shape[3]):
        fft_axis = np.arange(MSF1.shape[0]) * resolution
        plots[trial].plot(fft_axis + FFT_PARAMS["start_frequency"],
                                MSF1[:, canal-1, 0, trial, 0] , color = "#403e7d")
        plots[trial].set_xlabel('Frecuencia [Hz]')
        plots[trial].set_ylabel('Amplitud [uV]')
        # plots[trial].set_title(f'Estímulo {estim[0]} Hz del sujeto canal {canal}')
        plots[trial].xaxis.grid(True)
        plots[trial].axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
                                linestyle='--', color = "#e37165", alpha = 0.9)

plt.show()