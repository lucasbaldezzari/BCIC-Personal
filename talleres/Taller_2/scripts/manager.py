# -*- coding: utf-8 -*-
"""
manager

Created on Sat May  8 10:03:24 2021

@author: Lucas
"""

import fileAdmin as fa

import numpy as np

import matplotlib.pyplot as plt

from utils import plotEEG
from utils import filterEEG
from utils import segmentingEpochs, getSpectrum
from utils import plotSpectrum
# from utils import segmented_data, magnitude_spectrum
# from utils import plotSpectrumEEG

from scipy.fftpack import fft, fftfreq
import scipy.fftpack

path = "E:/reposBCICompetition/BCIC-Personal/Taller_2/scripts/dataset" #directorio donde estan los datos

sujetos = [1,2,3,4,5,6,7,8,9,10] #sujetos 1 y 2

setSubjects = fa.loadData(path = path, subjects = sujetos)

#Conociendo mis datos
print(type(setSubjects["s1"])) #tipo de datos del sujeto 1

print(setSubjects["s1"].keys()) #imprimimos las llaves del diccionario

print(setSubjects["s1"]["eeg"].shape) #imprimimos la forma del dato en "eeg"
# Obtenemos un arreglo que se corresponde con lo mencionado en la referencia
# [Number of targets, Number of channels, Number of sampling points, Number of trials]


"""Grafiquemos los datos obtenidos para el sujeto 1 en los 8 canales y el blanco de 9.25Hz"""


sujeto = 7
eeg = setSubjects[f"s{sujeto}"]["eeg"]
clases = eeg.shape[0] #clases del sujeto 1
channels = eeg.shape[1] #canales del sujeto 
samples= eeg.shape[2] #cantidad de muestras
trials= eeg.shape[3] # cantidad de trials

target = 7 #el máximo es 12

fm = 256.0

"""
Los autores del set de datos, citan:
    The onset of visual stimulation is at 39th sample point,
    which means there are redundant data for 0.15 [s] before stimulus onset.
    
Con esto en mente es que se descartan las primeras 39 muestras
"""

tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
muestraDescarte = 39

eeg = eeg[:,:, muestraDescarte: ,:]
eeg = eeg[:,:, :tiempoTotal ,:]

title = f"Señal de EEG sin filtro sujeto {sujeto}"
plotEEG(signal = eeg, sujeto = sujeto,
        trial = 3, blanco = target, window = [0,4], fm = 256.0, save = True, title = title)

#filtro la señal entre los 5hz y los 80hz
eegfiltrado = filterEEG(eeg, lfrec = 5., hfrec = 80., orden = 4, fm  = 256.0)

title = f"Señal de EEG filtrada sujeto {sujeto}"

plotEEG(eegfiltrado, sujeto = sujeto,
        trial = 3, blanco = target, window = [0,4], fm = 256.0, save = True,
        title = title)


################## Graficando espectro############################### 

espectroSujeto = dict()

fftpar = {
    'resolución': 0.25, #fm/cantidad de muestras
    'frecuencia inicio': 0.0,
    'frecuencia final': 35.0,
    'fm': 256.0
} #parámetros importantes para aplicar la FFT

frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])

ventana = 4
solapamiento = ventana*1
canal = 5

#Realizo la segmentación de mi señal de EEG con ventana y el solapamiento dados
eegSSegmentedo = segmentingEpochs(eegfiltrado, ventana, solapamiento, fm)

espectroSujeto[f"s{sujeto}"] = getSpectrum(eegSSegmentedo, fftpar)

#Grafico el espectro para todos los blancos para el canal propuesto
plotSpectrum(espectroSujeto[f"s{sujeto}"], fftpar["resolución"], clases, 
              sujeto, canal, frecStimulus, save = True)
