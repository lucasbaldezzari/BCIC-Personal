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

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

import os

from DataThread import DataThread as DT
from GraphModule import GraphModule as Graph
        

def main():
   
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
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
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
    board_shim.start_stream(450000, args.streamer_params)
    
    data_thread = DT(board_shim, args.board_id)
    graph = Graph(board_shim) 

    try:
        data_thread.start()
        graph.start() #init graphication
        while True:
            if keyboard.read_key() == "esc":
                print("Stopped by user")
                break
        
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
        
    finally:
        data_thread.keep_alive = False
        graph.keep_alive = False
        data_thread.join()
        graph.join()
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == "__main__":
        main()