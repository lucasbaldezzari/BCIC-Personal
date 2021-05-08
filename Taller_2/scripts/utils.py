# -*- coding: utf-8 -*-
"""
utils

Created on Sat May  8 10:31:22 2021

@author: Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
    
def plotEEG(signal, channelsNums, sujeto = 1, trial = 1, target = 1,
            fm = 256.0, window = 1.0, rmvOffset = False):
    
    
    T = 1.0/fm #período de la señal
    
    totalLenght = signal.shape[2]
    beginSample = window[0]
    endSample = window[1]    

    #Chequeo que la ventana de tiempo no supere el largo total
    if beginSample/T >= totalLenght or beginSample <0:
        beginSample = 0.0 #muevo el inicio de la ventana a 0 segundos
        
    if endSample/T >= totalLenght:
        endSample = totalLenght*T #asigno el largo total
        
    if (endSample - beginSample) >0:
        lenght = (endSample - beginSample)/T #cantidad de valores para el eje x
        t = np.arange(1,lenght)*T + beginSample
    else:
        lenght = totalLenght
        t = np.arange(1,lenght)*T #máxima ventana
        
    scaling = (5/2**16) #supongo Vref de 5V y un conversor de 16 bits
    signalAvg = 0
    
    #genero la grilla para graficar
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), gridspec_kw = dict(hspace=0.5, wspace=0.2))
    axes = axes.reshape(-1)
        
    for canal in range(channelsNums):
        if rmvOffset:
            signalAvg = np.average(signal[target][canal-1].T[trial-1][:len(t)])
        signalScale = (signal[target][canal-1].T[trial-1][:len(t)] - signalAvg)*scaling 
        axes[canal].plot(t, signalScale, color = "#e37165")
        axes[canal].set_xlabel('Tiempo [seg]') 
        axes[canal].set_ylabel('Amplitud [uV]')
        axes[canal].set_title(f'Sujeto {sujeto} - Blanco {target + 1} - Canal {canal + 1}')
        axes[canal].yaxis.grid(True)
        
    plt.title(f"Señal de EEG de sujeto {sujeto}")
    plt.show()