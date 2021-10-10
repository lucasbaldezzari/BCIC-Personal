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

import matplotlib.pyplot as plt

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG/LucasB")

trials = 10
fm = 200.
window = 5 #sec
samplePoints = int(fm*window)
channels = 4
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["lb-R2-S1-E7","lb-R1-S1-E9","lb-R2-S1-E11","lb-R2-S1-E13"]
allData = fa.loadData(path = path, filenames = filenames)

name = "lb-R2-S1-E7" #nombre de los datos a analizar
estim = [7] #Le pasamos un estímulo para que grafique una linea vertical

eeg = allData[name]['eeg']

#Chequeamos información del registro eeg 1
print(allData[name]["generalInformation"])
print(f"Forma de los datos {eeg.shape}")

#Filtramos la señal de eeg para eeg 1

resolution = np.round(fm/eeg.shape[2], 4)

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
trial = 10

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
#graficamos espectro para todos los trials y un canal
########################################################################

canal = 2 #elegimos un canal

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
        # plots[trial].legend()

plt.show()



# run1 = allData[names[0]]
# run2 = allData[names[1]]
# run3 = allData[names[2]]

# #Chequeamos información del registro eeg 1
# print(run2["generalInformation"])

# run1EEG = run2["eeg"]
# #[Number of targets, Number of channels, Number of sampling points, Number of trials]

# plotEEG(run1EEG, sujeto = 1, trial = 1, blanco = 1,
#             fm = fm, window = [0,5], rmvOffset = False, save = False, title = "", folder = "figs")

# resolution = np.round(fm/run1EEG.shape[2], 4)

# PRE_PROCES_PARAMS = {
#                 'lfrec': 5.,
#                 'hfrec': 28.,
#                 'order': 8,
#                 'sampling_rate': fm,
#                 'window': window,
#                 'shiftLen':window
#                 }

# FFT_PARAMS = {
#                 'resolution': resolution,
#                 'start_frequency': 0.,
#                 'end_frequency': 28.0,
#                 'sampling_rate': fm
#                 }

# run1EEGFiltered = filterEEG(run1EEG, PRE_PROCES_PARAMS["lfrec"],
#                         PRE_PROCES_PARAMS["hfrec"],
#                         PRE_PROCES_PARAMS["order"],
#                         PRE_PROCES_PARAMS["sampling_rate"])

# plotEEG(run1EEGFiltered, sujeto = 1, trial = 1, blanco = 1,
#             fm = fm, window = [0,5], rmvOffset = False, save = False,
#             title = "Señal de EEG filtrada del Sujeto 1", folder = "figs")


# #eeg data segmentation
# eegSegmented = segmentingEEG(run1EEGFiltered, PRE_PROCES_PARAMS["window"],
#                              PRE_PROCES_PARAMS["shiftLen"],
#                              PRE_PROCES_PARAMS["sampling_rate"])

# MSF = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)
# #(113, 8, 1, 3, 1)

# canal = 1
# estim = [13]
# plotOneSpectrum(MSF, resolution, 1, subjects[0], canal-1, estim,
#                 startFrecGraph = FFT_PARAMS['start_frequency'],
#               save = False, title = f"Espectro para canal  {canal}",
#               folder = "figs")

# #Graficamos espectro promediando canales
# import matplotlib.pyplot as plt
# trial = 10
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Amplitud [uV]')
# trial = 3
# plt.title(f"Espectro promediando los 4 para trial {trial} y estim {estim[0]}")
# fft_axis = np.arange(MSF.shape[0]) * resolution
# MSFmean = np.mean(MSF,axis = 1).reshape(MSF.shape[0],MSF.shape[3])
# plt.plot(fft_axis, MSFmean[:,trial-1])

# plt.axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
#                         label = f"Frecuencia estímulo {estim[0]}Hz",
#                         linestyle='--', color = "#e37165", alpha = 0.9)
# plt.legend()
# plt.show()

# #Graficamos espectro promediando canales
# import matplotlib.pyplot as plt
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Amplitud [uV]')
# plt.title(f"Espectro promediando los 4 y todos los trials - {estim[0]}")
# fft_axis = np.arange(MSF.shape[0]) * resolution
# MSFmean = np.mean(MSF,axis = 1).reshape(MSF.shape[0],MSF.shape[3])
# plt.plot(fft_axis, np.mean(MSFmean, axis = 1))

# plt.axvline(x = estim[0], ymin = 0., ymax = max(fft_axis),
#                         label = f"Frecuencia estímulo {estim[0]}Hz",
#                         linestyle='--', color = "#e37165", alpha = 0.9)
# plt.legend()
# plt.show()