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

# import matplotlib.pyplot as plt

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from ArduinoCommunication import ArduinoCommunication as AC

from DataThread import DataThread as DT
from SVMClassifier import SVMClassifier as SVMClassifier
from scipy.signal import windows

import fileAdmin as fa

def cargarClasificador(modelo, modelName, signalPSDName,frecStimulus, nsamples, path):
    """Cargamos clasificador que utilizaremos para clasificar nuestra señal"""

    #### TO DO: Agregar los clasificadores que faltan. ###

    actualFolder = os.getcwd()#directorio donde estamos actualmente
    os.chdir(path)

    if modelo == "SVM":
        modelName = "SVM_test_linear"
        modelFile = f"{modelName}.pkl" #nombre del modelo
        PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = path)

        clasificador = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS,
                            nsamples = nsamples, path = path) #cargamos clasificador entrenado
        
        clasificador.loadTrainingSignalPSD(filename = signalPSDName, path = path) #cargamos el PSD de mis datos de entrenamiento

    os.chdir(actualFolder)

    return clasificador

def clasificar(rawEEG, modelo, clasificador, canalesElegidos = 2, anchoVentana = 5, bw = 2., order = 6, axis = 0):

    #### TO DO: Agregar los clasificadores que faltan. ###
    
    rawEEG = rawEEG[:canalesElegidos,:]
    rawEEG = np.mean(rawEEG, axis = 0)
    if modelo == "SVM":
        featureVector = clasificador.extractFeatures(rawDATA = rawEEG, ventana = windows.hamming,
                        anchoVentana = anchoVentana, bw = bw, order = 4, axis = axis)
        comando = clasificador.getClassification(featureVector = featureVector)

    return comando

def main():

    """ ######################################################
    PASO 1: Cargamos datos generales de la sesión
    ######################################################"""

    actualFolder = os.getcwd() #Folder base
    path = os.path.join(actualFolder,"models")

    placas = {"cyton": BoardIds.CYTON_BOARD, #IMPORTANTE: frecuencia muestreo 256Hz
              "ganglion": BoardIds.GANGLION_BOARD, #IMPORTANTE: frecuencia muestro 200Hz
              "synthetic": BoardIds.SYNTHETIC_BOARD}
    
    placa = placas["ganglion"]

    trials = 5 #None implica que se ejecutaran trials de manera indeterminada
    trialDuration = 10 #secs #IMPORTANTE: trialDuration SIEMPRE debe ser MAYOR a stimuliDuration
    stimuliDuration = 5 #secs

    classifyData = True #True en caso de querer clasificar la señal de EEG
    fm = BoardShim.get_sampling_rate(placa)    
    window = stimuliDuration #segundos
    nsamples = int(fm*window)

    equipo = "neurorace"
    if equipo == "neurorace":
        frecStimulus = np.array([7, 9, 11, 13])
        listaEstims = frecStimulus.tolist()
        movements = [b'2',b'4',b'1',b'3',b'0']#izquierda, derecha, adelante, retroceso

    """ ##########################################################################################
    PASO 2: Cargamos datos necesarios para el clasificador y cargamos clasificador
    ##########################################################################################"""

    modelName = "SVM_test_linear"
    signalPSDName = "SVM_test_linear_signalPSD.txt"
    modeloClasificador = "SVM"
    clasificador = cargarClasificador(modelo = modeloClasificador, modelName = modelName, signalPSDName = signalPSDName,
                                    frecStimulus = frecStimulus, nsamples = nsamples, path = path)

    """ ##########################################################################################
    PASO 3: INICIO DE CARGA DE PARÁMETROS PARA PLACA OPENBCI
    Primeramente seteamos los datos necesarios para configurar la OpenBCI
    ##########################################################################################"""

    #First we need to load the Board using BrainFlow
   
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    
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

    fm = BoardShim.get_sampling_rate(args.board_id)
    """FIN DE CARGA DE PARÁMETROS PARA PLACA OPENBCI"""
    
    board_shim = BoardShim(args.board_id, params) #genero un objeto para control de placas de Brainflow
    board_shim.prepare_session()
    time.sleep(2) #esperamos 2 segundos

    """
    IMPORTANTE: No tocar estos parámetros.
    El string es:
    x (CHANNEL, POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET) X

    Doc: https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
    """

    if placa == BoardIds.CYTON_BOARD.value:
        board_shim.config_board("x1020110Xx2020110Xx3101000Xx4101000Xx5101000Xx6101000Xx7101000Xx8101000X")
        time.sleep(4)

    """ ##########################################################################################
    PASO 4: Iniciamos la recepción de dataos desde la placa OpenBCI
    ##########################################################################################"""
    
    board_shim.start_stream(450000, args.streamer_params) #iniciamos OpenBCI. Ahora estamos recibiendo datos.
    time.sleep(2)
    
    """genero un objeto DataThread para extraer datos de la OpenBCI"""
    data_thread = DT(board_shim, args.board_id)
    time.sleep(1)

    """ ##########################################################################################
    PASO 4: Inicio comunicación con Arduino instanciando un objeto AC (ArduinoCommunication)
    en el COM8, con un timing de 100ms

    - El objeto ArduinoCommunication generará una comunicación entre la PC y el Arduino
    una cantidad de veces dada por el parámetro "ntrials". Pasado estos n trials se finaliza la sesión.

    - En el caso de querer comunicar la PC y el Arduino por un tiempo indeterminado debe hacerse
    ntrials = None (default)
    ##########################################################################################"""

    #IMPORTANTE: Chequear en qué puerto esta conectado Arduino.
    arduino = AC('COM7', trialDuration = trialDuration, stimONTime = stimuliDuration,
             timing = 100, ntrials = trials)

    time.sleep(1) 
    arduino.iniSesion() #Inicio sesión en el Arduino.
    time.sleep(1) 

    try:
        while arduino.generalControl() == b"1":
            if classifyData and arduino.systemControl[1] == b"0":
                rawEEG = data_thread.getData(stimuliDuration)
                frecClasificada = clasificar(rawEEG, modeloClasificador, clasificador, canalesElegidos = 2, anchoVentana = 5, bw = 2., order = 6, axis = 0)
                print(f"Comando a enviar {movements[listaEstims.index(frecClasificada)]}. Frecuencia {frecClasificada}")
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