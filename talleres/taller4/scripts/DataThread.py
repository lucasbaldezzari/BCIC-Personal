"""
Created on Wed Jun 23 11:27:24 2021

@author: Lucas BALDEZZARI
"""

import time
import brainflow
import numpy as np
import threading

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

class DataThread(threading.Thread):
    
    def __init__(self, board, board_id):
        threading.Thread.__init__(self)
        
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.board = board
        
    def run (self):
        
        window_size = 4 #secs
        sleep_time = 1 #secs
        points_per_update = window_size * self.sampling_rate
        
        while self.keep_alive:
            time.sleep (sleep_time)
            # get current board data doesnt remove data from the buffer
            data = self.board.get_current_board_data(int(points_per_update))
            print(f"New data {data[:10]}")
            
def main():
    BoardShim.enable_dev_board_logger()
    
    # use synthetic board for demo
    params = BrainFlowInputParams ()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim (board_id, params)
    board.prepare_session ()
    board.start_stream ()
    
    data_thread = DataThread (board, board_id)
    data_thread.start ()
    
    try:
        time.sleep(4)
        
    finally:
        data_thread.keep_alive = False
        data_thread.join()
        
    board.stop_stream()
    board.release_session()
    
if __name__ == "__main__":
    main()
    
    
    
            