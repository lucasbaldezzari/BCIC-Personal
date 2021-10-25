import os
import numpy as np

import fileAdmin as fa
from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum, plotEEG
from utils import norm_mean_std

import matplotlib.pyplot as plt

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"recordedEEG")

trials = 1
fm = 250.
window = 4 #sec
samplePoints = int(fm*window)
channels = 8
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["lucasB_11hz_elecActivos","testeandoElectrodosActivos5"]
allData = fa.loadData(path = path, filenames = filenames)

eeg4 = allData["lucasB_11hz_elecActivos"]["eeg"]
eeg5 = allData["testeandoElectrodosActivos5"]["eeg"]


plotEEG(eeg4, sujeto = 1, trial = 1, blanco = 1,
            fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

eeg1filtered = filterEEG(eeg4, 4, 38, 8, 50., fm = 250.0)

plotEEG(eeg1filtered, sujeto = 1, trial = 1, blanco = 1,
            fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")

 plt.plot(eeg1filtered[0,0,:,0])           
 plt.show()