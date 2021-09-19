"""
Created on Wed Jun 23 11:27:24 2021

@author: Lucas BALDEZZARI

        VERSIÓN: SCT-01-RevA
"""

import time
import numpy as np
import threading

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
import os
from SVMClassifier import SVMClassifier as SVMClassifier


class DataThread(threading.Thread):
    
    def __init__(self, board, board_id):
        threading.Thread.__init__(self)
        
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.acel_channels = BoardShim.get_accel_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.board = board
        self.trialDuration = 4 #secs
        
    def getData (self, duration, channels = 8):

        data = self.board.get_current_board_data(int(duration*self.sampling_rate))
        eeg = []        
        eeg = [data[canal] for canal in self.eeg_channels]
        eeg = np.asarray(eeg)
        
        return eeg
        
        # return self.board.get_current_board_data(int(duration*self.sampling_rate))[:channels]
        
    def run (self):
        
        window_size = self.trialDuration
        sleep_time = 1 #secs
        points_per_update = window_size * self.sampling_rate
        
        while self.keep_alive:
            time.sleep (sleep_time)
            # get current board data doesnt remove data from the buffer
            data = self.board.get_current_board_data(int(points_per_update))
            # print(f"New data {data[:10]}")
            # print(f"New data {data.shape}")
            
def main():
    BoardShim.enable_dev_board_logger()
    
    # use synthetic board for demo only
    params = BrainFlowInputParams ()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim (board_id, params)
    board.prepare_session ()
    board.start_stream ()
    
    data_thread = DataThread(board, board_id) #Creo un  objeto del tipo DataThread
    data_thread.start () #Se ejecuta el método run() del objeto data_thread

    """
    Cargamos clasificador
    """

    path = "E:\reposBCICompetition\BCIC-Personal\scripts\Bases\models"
    
    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
    
    modelFile = "SVM14Channels.pkl"

    window = 4
    # fm = BoardShim.get_sampling_rate(board_id)
    fm = 256.
    samples = fm*window
    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'lfrec': 5.,
                    'hfrec': 38.,
                    'order': 4,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': 4,
                    'shiftLen':4
                    }
    
    resolution = np.round(fm/samples, 4)
    
    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 5.0,
                    'end_frequency': 38.0,
                    'sampling_rate': fm
                    }
                    
    frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])    
    svm = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, path = path)
    
    try:
        time.sleep(5)
        rawEEG = data_thread.getData(4)
        print(rawEEG.shape)
        frecClasificada = svm.getClassification(rawEEG = rawEEG[:8, :])
        print(f"El estímulo clasificado fue {frecClasificada}")
        # print(currentData[:4, :].shape)
        
    finally:
        data_thread.keep_alive = False
        data_thread.join() #free thread
        
    board.stop_stream()
    board.release_session()
    
if __name__ == "__main__":
    main()
    
    
    
            