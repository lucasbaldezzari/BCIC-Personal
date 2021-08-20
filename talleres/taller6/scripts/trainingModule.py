# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:57:43 2021

@author: Lucas Baldezzari


Módulo de control utilizado para adquirir y almacenar datos de EEG.

Los procesos principales son:
    - Seteo de parámetros y conexión con placa OpenBCI (Synthetic, Cyton o Ganglion)
    para adquirir datos en tiempo real.
    - Comunicación con placa Arduino para control de estímulos.
    - Adquisición de señales de EEG a partir de la placa OpenBCI.
    - Control de trials: Pasado ntrials se finaliza la sesión.
    - Registro de EEG: Finalizada la sesión se guardan los datos con saveData() de fileAdmin

"""

import os
import argparse
import time
import logging
import numpy as np
import threading
# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from ArduinoCommunication import ArduinoCommunication as AC

from DataThread import DataThread as DT
# from GraphModule import GraphModule as Graph       
import fileAdmin as fa

def main():
    
    """INICIO DE CARGA DE PARÁMETROS PARA PLACA OPENBCI"""
    """Primeramente seteamos los datos necesarios para configurar la OpenBCI"""
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

    #IMPORTENTE: Chequear en que puerto esta conectada la OpenBCI. En este ejemplo esta en el COM4    
    # parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM4')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    # parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #                     required=False, default=BoardIds.CYTON_BOARD)
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
    
    """FIN DE CARGA DE PARÁMETROS PARA PLACA OPENBCI"""
    
    board_shim = BoardShim(args.board_id, params) #genero un objeto para control de placas de Brainflow
    board_shim.prepare_session()
    time.sleep(2) #esperamos 2 segundos
    
    board_shim.start_stream(450000, args.streamer_params) #iniciamos OpenBCI. Ahora estamos recibiendo datos.
    time.sleep(4) #esperamos 4 segundos
    
    data_thread = DT(board_shim, args.board_id) #genero un objeto DataThread para extraer datos de la OpenBCI
    time.sleep(1)

    """Defino variables para control de Trials"""
    
    trials = 2 #cantidad de trials. Sirve para la sesión de entrenamiento.
    #IMPORTANTE: trialDuration SIEMPRE debe ser MAYOR a stimuliDuration
    trialDuration = 8 #secs
    stimuliDuration = 4 #secs

    saveData = True
    
    EEGdata = []
    fm = 250
    
    samplePoints = int(fm*stimuliDuration)
    channels = 8
    stimuli = 1 #one stimulus
    
    """Inicio comunicación con Arduino instanciando un objeto AC (ArduinoCommunication)
    en el COM3, con un timing de 100ms
    
    - El objeto ArduinoCommunication generará una comunicación entre la PC y el Arduino
    una cantidad de veces dada por el parámetro "ntrials". Pasado estos n trials se finaliza la sesión.
    
    - En el caso de querer comunicar la PC y el Arduino por un tiempo indeterminado debe hacerse
    ntrials = None (default)
    """
    #IMPORTANTE: Chequear en qué puerto esta conectado Arduino.
    #En este ejemplo esta conectada en el COM3
    ard = AC('COM3', trialDuration = trialDuration, stimONTime = stimuliDuration,
             timing = 100, ntrials = trials)
    time.sleep(2) 
    
    path = "recordedEEG" #directorio donde se almacenan los registros de EEG.
    
    #El siguiente diccionario se usa para guardar información relevante cómo así también los datos de EEG
    #registrados durante la sesión de entrenamiento.
    dictionary = {
                'subject': 'Test2',
                'date': '19/08/2021',
                'generalInformation': 'Test',
                "channels": "[1,2,3,4,5,6,7,8]", 
                 'dataShape': [stimuli, channels, samplePoints, trials],
                  'eeg': None
                    }

    ard.iniSesion() #Inicio sesión en el Arduino.
    # graph = Graph(board_shim)

    try:
        # data_thread.start()
        # graph.start() #init graphication
        # while True:
        #     if keyboard.read_key() == "esc":
        #         print("Stopped by user")
        #         break
    
        while ard.generalControl() == b"1":
            if saveData and ard.systemControl[1] == b"0":
                currentData = data_thread.getData(stimuliDuration)
                EEGdata.append(currentData)
                saveData = False
            elif saveData == False and ard.systemControl[1] == b"1":
                saveData = True
        
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
            
        #ard.endSesion() #finalizo sesión (se apagan los estímulos)
        ard.close() #cierro comunicación serie para liberar puerto COM
        
        #Guardo los datos registrados por la placa
        EEGdata = np.asarray(EEGdata)
        rawEEG = EEGdata.reshape(1,EEGdata.shape[0],EEGdata.shape[1],EEGdata.shape[2])
        rawEEG = rawEEG.swapaxes(1,2).swapaxes(2,3)
        dictionary["eeg"] = rawEEG
        fa.saveData(path = path,dictionary = dictionary, fileName = dictionary["subject"])

if __name__ == "__main__":
        main()