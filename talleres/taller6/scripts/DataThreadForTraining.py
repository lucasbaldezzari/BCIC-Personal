"""
Created on Wed Jun 23 11:27:24 2021

@author: Lucas BALDEZZARI
"""

import os
import time
import brainflow
import numpy as np
import threading
import keyboard

import fileAdmin as fa

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum

class DataThread(threading.Thread):
    
    def __init__(self, board, board_id, dictionary = dict()):
        
        threading.Thread.__init__(self)
        
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.board = board
        
        self.dataShape = dictionary["dataShape"]
        
        self.channels = [1]
        self.samplePoints = [2]
        self.trials = self.dataShape[3]
        
        self.eegData = list()

        
    def run (self):
        
        window_size = 4 #secs
        sleep_time = 1 #secs
        points_per_update = window_size * self.sampling_rate
        
        waitingTime = 5 #secs
        counter = int(waitingTime/sleep_time)
        
        timer = 0
        trial = 0
        
        while counter > 0:
            counter -= counter
            print(f"Cuenta regresiva {counter}")
            time.sleep(sleep_time)
        
        while self.keep_alive:
            if timer == 4:
                # timer = 0
                data = self.board.get_current_board_data(int(points_per_update))[6:14]
                self.eegData.append(data)
            if timer == 5:
                timer = 0
                trial += 1
                print(f"Trial número {trial}")
                if trial == self.trials:
                    self.keep_alive = False
                
            timer +=1            
            time.sleep(sleep_time)
            
        print("Training session finished. Please press Escape.")
            
def main():
    
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
    
    BoardShim.enable_dev_board_logger()
    
    # use synthetic board for demo only
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    data_thread = DataThread(board, board_id, dictionary) #Creo un  objeto del tipo DataThread
    data_thread.starts() #Se ejecuta el método run() del objeto data_thread
    
    try:
        while True:
            if keyboard.read_key() == "esc":
                print("Stopped by user")
                break
        
    finally:
        data_thread.keep_alive = False
        data_thread.join() #free thread
        
        eegData = np.asarray(data_thread.eegData)
        rawEEG = eegData.reshape(1,eegData.shape[0],eegData.shape[1],eegData.shape[2])
        rawEEG = rawEEG.swapaxes(1,2).swapaxes(2,3)
        dictionary["eeg"] = rawEEG
        fa.saveData(path = path, dictionary = dictionary, fileName = dictionary["subject"])
        
    board.stop_stream()
    board.release_session()
    
if __name__ == "__main__":
    main()
    


    
            