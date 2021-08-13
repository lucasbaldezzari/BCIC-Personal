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

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum

from brainflow.data_filter import DataFilter

def loadData(path ="/dataset", filenames = ["s1"]):
    setSubjects = {}
    for filename in filenames:
        
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
        
        dataSet = sciio.loadmat(f"{path}\{filename}.mat")
        
        dataSet["eeg"] = np.array(dataSet['eeg'], dtype='float32') #convierto los datos a flotantes
        
        setSubjects[f"{filename}"] = dataSet #guardo los datos para el sujeto correspondiente
        
    return setSubjects

def saveData(path ="/recordedEEG", dictionary = dict(), fileName = "subject1"):
    
    """
    Save the EEG data into a mat file
    
    Args:
        - path: folder where we'll save the file
        - dictionary: A dictionary with shape,
            {
            'subject': 'A string for subject'
            'date': 'a date in string',
            'generalInformation': 'An string with general information',
             'dataShape': 'a list with the EEG data shape, like [number of stimuli, channels, samples, trials]',
              'EEGData': numpy.ndarray with the EEG data with shape like dataShape  
                }
    """
    
    sciio.savemat(f"{path}\{fileName}.mat", dictionary)

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
    dataSet["eeg"] = np.array(dataSet['eeg']) #convierto los datos a flotantes
    eeg = dataSet['eeg'][target-1,:,:,trial-1]
    data = np.zeros((eeg.shape[0]*4+1,eeg.shape[1]))
    eeg = np.repeat(eeg,4,axis = 0)
    
    DataFilter.write_file(eeg, filename, 'w')
    
    
    samples = np.arange(0,eeg.shape[1])
    # data[0] = samples
    # data[1:] = eeg
    # np.savetxt(filename, data.T, delimiter=" ")
    
def main():
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"dataset")
    
    subjects = [2]
        
    fm = 256.0
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    frecStimulus = np.array([9.25, 11.25, 13.25,
                              9.75, 11.75, 13.75,
                              10.25, 12.25, 14.25,
                              10.75, 12.75, 14.75])
    
    """
    **********************************************************************
    Loading and plotting the EEG
    IMPORTANT: In real time BCI, the "loading data" will be replaced
    for real data coming from the OpenBCI board
    **********************************************************************
    """
    allData = loadData(path = path, subjects = subjects)
    rawEEG = loadData(path = path, subjects = subjects)[f"s{subjects[0]}"]["eeg"]
    
    dictionary = {
                'subject': 's1',
                'date': '27 de julio',
                'generalInformation': 'Primer ensayo',
                 'dataShape': [1, 8, 1000, 1],
                  'eeg': rawEEG 
                    }
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG")
    
    saveData(path = path, dictionary = dictionary, fileName = "s2")
    
    eeg = loadData(path = path, subjects = subjects)[f"s{subjects[0]}"]["eeg"]
    
# if __name__ == "__main__":
#     main()



