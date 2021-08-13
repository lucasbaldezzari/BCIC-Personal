# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:57:43 2021

@author: Lucas Baldezzari

Principal module in orer to ejecute the entire system.

- Signal Processing (SP) module
- Classification (CLASS) module
- Graphication (GRAPH) module
"""

import argparse
import time
import logging
import numpy as np
import threading
import keyboard
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from ArduinoCommunication import ArduinoCommunication as AC

import os

from DataThread import DataThread as DT
from GraphModule import GraphModule as Graph       
import fileAdmin as fa

def main():
    
    #First we need to load the Board using BrainFlow
   
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    # parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM4')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    # parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #                     required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.CYTON_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    
    board_shim = BoardShim(args.board_id, params)
    board_shim.prepare_session()
    time.sleep(2) #esperamos 2 segundos
    
    board_shim.start_stream(450000, args.streamer_params) #iniciamos board
    time.sleep(4)
    
    data_thread = DT(board_shim, args.board_id)
    time.sleep(1)
    
    trials = 15
    ard = AC('COM3', timing = 500, trials = trials)
    time.sleep(2) 
    
    ard.iniSesion()
    
    trialDuration = 4 #secs
    oldTrial = -1
    actualTrial = ard.trial
    saveData = True
    
    EEGdata = []
    fm = 250.
    
    samplePoints = int(fm*trialDuration)
    channels = 8
    stimuli = 1 #one stimulus
    
    path = "recordedEEG"
    
    dictionary = {
                'subject': 'LucasB-PruebaSSVEPs(11Hz)-Num1',
                'date': '12/08/2021',
                'generalInformation': 'Datos desde Cyton. Testeando SSVEPs en 11Hz. Descartar trial 1',
                "channels": "[1,2,3,4]", 
                 'dataShape': [stimuli, channels, samplePoints, trials],
                  'eeg': None
                    }
    # graph = Graph(board_shim)

    try:
        # data_thread.start()
        # graph.start() #init graphication
        # while True:
        #     if keyboard.read_key() == "esc":
        #         print("Stopped by user")
        #         break
    
        while ard.generalControl() == b"1":
            # if ard.trial-1 == 2:
            #     ard.endSesion()
            if saveData and ard.stimuliState[1] == b"0":
                currentData = data_thread.getData(trialDuration)
                EEGdata.append(currentData)
                saveData = False
            elif saveData == False and  ard.stimuliState[1] == b"1":
                saveData = True
            # if actualTrial != ard.trial: #tenemos nuevo trial
            # pass
        
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
        
    finally:
        # data_thread.keep_alive = False
        # graph.keep_alive = False
        # data_thread.join()
        # graph.join()
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
        ard.endSesion()    
        ard.close()
        
        EEGdata = np.asarray(EEGdata)
        rawEEG = EEGdata.reshape(1,EEGdata.shape[0],EEGdata.shape[1],EEGdata.shape[2])
        rawEEG = rawEEG.swapaxes(1,2).swapaxes(2,3)
        dictionary["eeg"] = rawEEG
        fa.saveData(path = path, dictionary = dictionary, fileName = dictionary["subject"])
        
        # print(rawEEG.shape)
        # plt.plot(rawEEG[0,0,:250,0])
        # plt.plot(rawEEG[0,1,:250,0])
        # plt.plot(rawEEG[0,2,:,0])
        # plt.plot(rawEEG[0,3,:,0])
        # plt.plot(rawEEG[0,4,:,0])
        # plt.show()


if __name__ == "__main__":
        main()