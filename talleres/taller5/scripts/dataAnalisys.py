# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:05:39 2021

@author: Lucas
"""
import os
import numpy as np

import fileAdmin as fa
from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum, plotEEG

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG")

trials = 15
fm = 250.
window = 4 #sec
samplePoints = int(fm*window)
channels = 8
stimuli = 1 #one stimulus

dictionary = {
            'subject': 's1',
            'date': '28 de julio',
            'generalInformation': 'Datos obtenidos desde la sintetic board.\
                Son datos de prueba.',
             'dataShape': [stimuli, channels, samplePoints, trials],
              'eeg': None
                }

subjects = [1]
eeg = fa.loadData(path = path, subjects = subjects)[f"{dictionary['subject']}"]["eeg"]

plotEEG(eeg, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

resolution = fm/eeg.shape[2]

PRE_PROCES_PARAMS = {
                'lfrec': 3.,
                'hfrec': 38.,
                'order': 4,
                'sampling_rate': fm,
                'window': 4,
                'shiftLen':4
                }

FFT_PARAMS = {
                'resolution': resolution,
                'start_frequency': 5.0,
                'end_frequency': 38.0,
                'sampling_rate': fm
                }

# #NOTA IMPORTANTE: Los datos provenientes de la Synthetic board NO necesitan ser filtrados
# filteredEEG = filterEEG(eeg, PRE_PROCES_PARAMS["lfrec"],
#                         PRE_PROCES_PARAMS["hfrec"],
#                         PRE_PROCES_PARAMS["order"],
#                         PRE_PROCES_PARAMS["sampling_rate"])

plotEEG(eeg, sujeto = 1, trial = 1, blanco = 1,
            fm = fm, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")



#eeg data segmentation
eegSegmented = segmentingEEG(eeg, PRE_PROCES_PARAMS["window"],
                             PRE_PROCES_PARAMS["shiftLen"],
                             PRE_PROCES_PARAMS["sampling_rate"])

MSF = computeMagnitudSpectrum(eegSegmented, FFT_PARAMS)

plotOneSpectrum(MSF, resolution, 1, subjects[0], 5, [14.75],
              startFrecGraph = FFT_PARAMS['start_frequency'],
              save = False,
              title = f"Espectro de los canales 8 a 16 -filtrados- de la Synthetic Board",
              folder = "figs")