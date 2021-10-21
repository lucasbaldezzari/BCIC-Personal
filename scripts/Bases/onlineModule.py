# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 2021

@author: Lucas Baldezzari

Módulo para adquirir, procesar y clasificar señales de EEG en busca de SSVEPs para obtener un comando

Los procesos principales son:
    - Seteo de parámetros y conexión con placa OpenBCI (Synthetic, Cyton o Ganglion)
    para adquirir datos en tiempo real.
    - Comunicación con placa Arduino para control de estímulos.
    - Adquisición de señales de EEG a partir de la placa OpenBCI.
    - Control de trials: Pasado ntrials se finaliza la sesión.
    - Sobre el EEG: Procesamiento, extracción de características, clasificación y obtención de un comando (traducción)
    
    *********** VERSIÓN: SCT-01-RevA ***********
    
    Funcionalidades:
        - Comunicación con las boards Cyton, Ganglion y Synthetic de OpenBCI
        - Comunicación con Arduino
        - Control de trials
        - Procesamiento, extracción de características, clasificación y obtención de un comando (traducción)
        - Actualización de variables de estado que se envían al Arduino M1

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
from GraphModule import GraphModule as Graph       
import fileAdmin as fa
from SVMClassifier import SVMClassifier as SVMClassifier

def main():
    
    """INICIO DE CARGA DE PARÁMETROS PARA PLACA OPENBCI"""
    """Primeramente seteamos los datos necesarios para configurar la OpenBCI"""
    #First we need to load the Board using BrainFlow
   
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    
    placas = {"cyton": BoardIds.CYTON_BOARD, #IMPORTANTE: frecuencia muestreo 256Hz
              "ganglion": BoardIds.GANGLION_BOARD, #IMPORTANTE: frecuencia muestro 200Hz
              "synthetic": BoardIds.SYNTHETIC_BOARD}
    
    placa = placas["ganglion"]  
    
    puerto = "COM5" #Chequear el puerto al cual se conectará la placa
    
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
    time.sleep(2) #esperamos 2 segundos
    
    board_shim.start_stream(450000, args.streamer_params) #iniciamos OpenBCI. Ahora estamos recibiendo datos.
    time.sleep(4) #esperamos 4 segundos
    
    data_thread = DT(board_shim, args.board_id) #genero un objeto DataThread para extraer datos de la OpenBCI
    time.sleep(1)

    """Defino variables para control de Trials"""
    
    trials = 1 #None implica que se ejecutaran trials de manera indeterminada
    trialDuration = 7 #secs #IMPORTANTE: trialDuration SIEMPRE debe ser MAYOR a stimuliDuration
    stimuliDuration = 5 #secs
    
    EEGdata = []

    stimuli = 1 #one stimulus

    """
    Cargamos clasificador
    """
    equipo = "neurorace"
    if equipo == "neurorace":
        frecStimulus = np.array([7, 9, 11, 13])
        listaEstims = frecStimulus.tolist()
        movements = [b'2',b'4',b'1',b'3',b'0']#izquierda, derecha, adelante, retroceso

    classifyData = True
    fm = BoardShim.get_sampling_rate(args.board_id)
    window = stimuliDuration #sec
    samplePoints = int(fm*window)
    channels = 4

    path = os.path.join('E:\\reposBCICompetition\\BCIC-Personal\\scripts\\Bases',"models")
    modelFile = "Logreg_LucasB_Test2_10112021.pkl" #nombre del modelo

    

    PRE_PROCES_PARAMS = {
                    'lfrec': 5.,
                    'hfrec': 38.,
                    'order': 8,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': window,
                    'shiftLen':window
                    }

    resolution = np.round(fm/samplePoints, 4)

    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 5.0,
                    'end_frequency': 38.0,
                    'sampling_rate': fm
                    }

    svm = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS, path = path)
    
    """Inicio comunicación con Arduino instanciando un objeto AC (ArduinoCommunication)
    en el COM8, con un timing de 100ms
    
    - El objeto ArduinoCommunication generará una comunicación entre la PC y el Arduino
    una cantidad de veces dada por el parámetro "ntrials". Pasado estos n trials se finaliza la sesión.
    
    - En el caso de querer comunicar la PC y el Arduino por un tiempo indeterminado debe hacerse
    ntrials = None (default)
    """
    #IMPORTANTE: Chequear en qué puerto esta conectado Arduino.
    arduino = AC('COM7', trialDuration = trialDuration, stimONTime = stimuliDuration,
             timing = 100, ntrials = trials)
    time.sleep(2) 

    arduino.iniSesion() #Inicio sesión en el Arduino.

    time.sleep(1) 

    try:
        while arduino.generalControl() == b"1":
            if classifyData and arduino.systemControl[1] == b"0":
                rawEEG = data_thread.getData(stimuliDuration)
                frecClasificada = svm.getClassification(rawEEG = rawEEG)
                print(f"Frecuencia clasificada {frecClasificada}")
                print(f"Comando a enviar {movements[listaEstims.index(frecClasificada)]}")
                arduino.systemControl[2] = b"1"#movements[listaEstims.index(int(frecClasificada))]#movements[3]
                esadoRobot = arduino.sendMessage(arduino.systemControl)
                classifyData = False
            elif classifyData == False and arduino.systemControl[1] == b"1":
                # arduino.systemControl[2] = movements[3]
                classifyData = True
        
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
        
    finally:
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
            
        arduino.close() #cierro comunicación serie para liberar puerto COM
        
if __name__ == "__main__":
        main()