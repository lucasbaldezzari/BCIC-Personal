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
window = 5 #sec
samplePoints = int(fm*window)
channels = 8
stimuli = 1 #one stimulus

subjects = [1]
filenames = ["testeandoElectrodosActivos","testeandoElectrodosActivos2"]
allData = fa.loadData(path = path, filenames = filenames)

eeg1 = allData["testeandoElectrodosActivos"]["eeg"]
eeg2 = allData["testeandoElectrodosActivos2"]["eeg"]


plotEEG(eeg2, sujeto = 1, trial = 1, blanco = 1,
            fm = 250.0, window = [0,4], rmvOffset = False, save = False, title = "", folder = "figs")