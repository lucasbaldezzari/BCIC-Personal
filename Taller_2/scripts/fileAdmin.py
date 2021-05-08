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
        Referencia: A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials
        
        Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject.
        Please see the reference paper for the detail.
        
        [Number of targets, Number of channels, Number of sampling points, Number of trials] = size(eeg)
        
        Number of targets : 12
        Number of channels : 8
        Number of sampling points : 1114
        Number of trials : 15
        Sampling rate [Hz] : 256
        """
        
        dataSet = sciio.loadmat(f"{path}/s{subject}.mat")
        
        dataSet["eeg"] = np.array(dataSet['eeg'], dtype='float32') #convierto los datos a flotantes
        
        setSubjects[f"s{subject}"] = dataSet #guardo los datos para el sujeto correspondiente
        
    return setSubjects
        
        # print(len(dataSet["train"][0][0]))
    
        # eeg = np.array(dataset['eeg'], dtype='float32')
        
        # num_classes = eeg.shape[0]
        # n_ch = eeg.shape[1]
        # total_trial_len = eeg.shape[2]
        # num_trials = eeg.shape[3]
        
        # filtered_data = su.get_filtered_eeg(eeg, 6, 80, 4, sample_rate)
        # segmented_data = su.get_segmented_epochs(filtered_data, window_len, shift_len, sample_rate)
        # magnitude_spectrum[f's{subject+1}'] = su.magnitude_spectrum_features(segmented_data, FFT_PARAMS)







