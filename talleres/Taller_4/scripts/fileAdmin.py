# -*- coding: utf-8 -*-
"""

fileAdmin

Created on Fri May  7 19:05:10 2021

@author: Lucas
"""

import sys
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io as sciio #Para manejar archivos de matlab .mat

def loadData(path ="/dataset", subjects = [1]):
    setSubjects = {}
    for subject in subjects:
        
        """
        Referencia: "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials"
        
        Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject.
        Please see the reference paper for the detail.
        
        [Number of targets, Number of channels, Number of sampling points, Number of trials] = size(eeg)
        
        Number of targets : 12
        Number of channels : 8
        Number of sampling points : 1114
        Number of trials : 15
        Sampling rate [Hz] : 256
        """
        
        dataSet = sciio.loadmat(f"{path}\s{subject}.mat")
        
        dataSet["eeg"] = np.array(dataSet['eeg'], dtype='float32') #convierto los datos a flotantes
        
        setSubjects[f"s{subject}"] = dataSet #guardo los datos para el sujeto correspondiente
        
    return setSubjects

def csvGenerator(path ="dataset", subject = 2, trial = 1, target = 1, filename = "eeg.csv"):
    
    #genero mi archivo csv
    
    """
    Referencia: "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials"
    
    Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject.
    Please see the reference paper for the detail.
    
    [Number of targets, Number of channels, Number of sampling points, Number of trials] = size(eeg)
    
    Number of targets : 12
    Number of channels : 8
    Number of sampling points : 1114
    Number of trials : 15
    Sampling rate [Hz] : 256
    """
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder, path)
    
    dataSet = sciio.loadmat(f"{path}\s{subject}.mat")
    dataSet["eeg"] = np.array(dataSet['eeg'], dtype='float32') #convierto los datos a flotantes
    eeg = dataSet['eeg'][target-1,:,:,trial-1]
    data = np.zeros((eeg.shape[0]+1,eeg.shape[1]))
    samples = np.arange(0,eeg.shape[1])
    data[0] = samples
    data[1:] = eeg
    np.savetxt(filename, data.T, delimiter=" ")





