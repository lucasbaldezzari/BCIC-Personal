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

path = "E:/reposBCICompetition/BCIC-Personal/Taller_2/scripts/dataset" #directorio donde estan los datos
sujetos = [1,2] #sujetos 1 y 2

setSubjects = fa.loadData(path = path, subjects = sujetos)

#Conociendo mis datos
print(type(setSubjects["s1"])) #tipo de datos del sujeto 1

print(setSubjects["s1"].keys()) #imprimimos las llaves del diccionario

print(setSubjects["s1"]["eeg"].shape) #imprimimos la forma del dato en "eeg"
# Obtenemos un arreglo que se corresponde con lo mencionado en la referencia
# [Number of targets, Number of channels, Number of sampling points, Number of trials]


"""Grafiquemos los datos obtenidos para el sujeto 1 en los 8 canales y el blanco de 9.25Hz"""
sujeto = 1
eegS1 = setSubjects["s1"]["eeg"]
clasesS1 = eegS1.shape[0] #clases del sujeto 1
channelsS1 = eegS1.shape[1] #canales del sujeto 
samplesS1= eegS1.shape[2] #cantidad de muestras
trialsS1= eegS1.shape[3] # cantidad de trials

fm = 256.0
    
plotEEG(signal = eegS1, channelsNums = channelsS1, sujeto = sujeto,
        trial = 3, target = 1, window = [0.1,0.5], fm = 256.0)