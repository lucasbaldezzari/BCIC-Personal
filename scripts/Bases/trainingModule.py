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
    
    VERSIÓN: SCT-01-RevB
    
    Funcionalidades:
        - Comunicación con las boards Cyton, Ganglion y Synthetic de OpenBCI
        - Comunicación con Arduino
        - Control de trials
        - Registro de datos adquiridos durante la sesión de entrenamiento.

"""

import argparse
import time
import logging
import numpy as np
# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from ArduinoCommunication import ArduinoCommunication as AC

from DataThread import DataThread as DT    
import fileAdmin as fa

def main():
    
    """INICIO DE CARGA DE PARÁMETROS PARA PLACA OPENBCI"""
    """Primeramente seteamos los datos necesarios para configurar la OpenBCI"""
    #First we need to load the Board using BrainFlow
   
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    
    placas = {"cyton": BoardIds.CYTON_BOARD.value, #IMPORTANTE: frecuencia muestreo 256Hz
              "ganglion": BoardIds.GANGLION_BOARD.value, #IMPORTANTE: frecuencia muestro 200Hz
              "synthetic": BoardIds.SYNTHETIC_BOARD.value}
    
    placa = placas["cyton"]  
    electrodos = "activos"
    
    puerto = "COM7" #Chequear el puerto al cual se conectará la placa
    
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
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default = puerto)
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default = placa)
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
    time.sleep(1) #esperamos 2 segundos

    #### CONFIGURAMOS LA PLACA CYTON ######
    """
    IMPORTANTE: No tocar estos parámetros.
    El string es:
    x (CHANNEL, POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET) X

    Doc: https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
    """

    if electrodos == "pasivos":
        configCanalesCyton = {
            "canal1": "x1060110X", #ON|Ganancia 24x|Normal input|Connect from Bias|
            "canal2": "x2060110X", #ON|Ganancia 24x|Normal input|Connect from Bias|
            "canal3": "x3101000X", #Canal OFF
            "canal4": "x4101000X", #Canal OFF
            "canal5": "x5101000X", #Canal OFF
            "canal6": "x6101000X", #Canal OFF
            "canal7": "x7101000X", #Canal OFF
            "canal8": "x8101000X", #Canal OFF
        }
        for config in configCanalesCyton:
            board_shim.config_board(configCanalesCyton[config])
            time.sleep(0.5)

    if electrodos == "activos":
        configCanalesCyton = {
            "canal1": "x1040110X", #ON|Ganancia 8x|Normal input|Connect from Bias|
            "canal2": "x2040110X", #ON|Ganancia 8x|Normal input|Connect from Bias|
            "canal3": "x3101000X", #Canal OFF
            "canal4": "x4101000X", #Canal OFF
            "canal5": "x5101000X", #Canal OFF
            "canal6": "x6101000X", #Canal OFF
            "canal7": "x7101000X", #Canal OFF
            "canal8": "x8101000X", #Canal OFF
        }
        for config in configCanalesCyton:
            board_shim.config_board(configCanalesCyton[config])
            time.sleep(0.5)

    board_shim.start_stream(450000, args.streamer_params) #iniciamos OpenBCI. Ahora estamos recibiendo datos.
    time.sleep(2) #esperamos 4 segundos
    
    data_thread = DT(board_shim, args.board_id) #genero un objeto DataThread para extraer datos de la OpenBCI
    time.sleep(1)

    """Defino variables para control de Trials"""
    
    trials = 5 #cantidad de trials. Sirve para la sesión de entrenamiento.
    #IMPORTANTE: trialDuration SIEMPRE debe ser MAYOR a stimuliDuration
    trialDuration = 10 #secs
    stimuliDuration = 5 #secs

    saveData = True
    
    EEGdata = []
    fm = BoardShim.get_sampling_rate(args.board_id)
    # channels = 8
    channels = len(BoardShim.get_eeg_channels(args.board_id))
    samplePoints = int(fm*stimuliDuration)
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
    arduino = AC('COM8', trialDuration = trialDuration, stimONTime = stimuliDuration,
             timing = 100, ntrials = trials)
    time.sleep(1) 
    
    path = "recordedEEG" #directorio donde se almacenan los registros de EEG.
    
    #El siguiente diccionario se usa para guardar información relevante cómo así también los datos de EEG
    #registrados durante la sesión de entrenamiento.
    dictionary = {
                'subject': 'lucasB_leds_20hz_g14_pasivos',
                'date': '5/11/2021',
                'generalInformation': 'Cyton. Estimulos LED. Ganancia en 8',
                'stimFrec': "20",
                'channels': [1,2,3,4], 
                 'dataShape': [stimuli, channels, samplePoints, trials],
                  'eeg': None
                    }

    arduino.iniSesion() #Inicio sesión en el Arduino.
    # graph = Graph(board_shim)
    time.sleep(1) 
    arduino.systemControl[2] = arduino.movements[3] #comando número 4 (b'3') [b'0',b'1',b'2',b'3',b'4',b'5']
    try:
        while arduino.generalControl() == b"1":
            if saveData and arduino.systemControl[1] == b"0":
                currentData = data_thread.getData(stimuliDuration, channels = channels)
                EEGdata.append(currentData)
                saveData = False
            elif saveData == False and arduino.systemControl[1] == b"1":
                saveData = True
        
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
        
    finally:
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
            
        arduino.close() #cierro comunicación serie para liberar puerto COM
        
        #Guardo los datos registrados por la placa
        EEGdata = np.asarray(EEGdata)
        rawEEG = EEGdata.reshape(1,EEGdata.shape[0],EEGdata.shape[1],EEGdata.shape[2])
        rawEEG = rawEEG.swapaxes(1,2).swapaxes(2,3)
        dictionary["eeg"] = rawEEG
        fa.saveData(path = path,dictionary = dictionary, fileName = dictionary["subject"])
if __name__ == "__main__":
        main()
        
        