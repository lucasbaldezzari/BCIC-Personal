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

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from ArduinoCommunication import ArduinoCommunication as AC

from DataThread import DataThread as DT
from SVMClassifier import SVMClassifier as SVMClassifier
from CNNClassifier import CNNClassifier

from scipy.signal import windows

import fileAdmin as fa

from GraphModule import GraphModule as Graph

def cargarClasificador(modelo, modelName, signalPSDName,frecStimulus, nsamples, path):
    """Cargamos clasificador que utilizaremos para clasificar nuestra señal"""

    #### TO DO: Agregar los clasificadores que faltan. ###

    actualFolder = os.getcwd()#directorio donde estamos actualmente
    os.chdir(path)

    if modelo == "SVM":
        modelName = modelName
        modelFile = f"{modelName}.pkl" #nombre del modelo
        PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = path)

        clasificador = SVMClassifier(modelFile, frecStimulus, PRE_PROCES_PARAMS, FFT_PARAMS,
                            nsamples = nsamples, path = path) #cargamos clasificador entrenado

        clasificador.loadTrainingSignalPSD(filename = signalPSDName, path = path) #cargamos el PSD de mis datos de entrenamiento

    if modelo == "CNN":
        path = os.path.join(actualFolder,path)
        os.chdir(path)
        weightFile = f'bestWeightss_{modelName}'
        PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = path)
        clasificador = CNNClassifier(modelFile = modelName,
                        weightFile = weightFile,
                        frecStimulus = frecStimulus.tolist(),
                        nchannels = 1, nsamples = nsamples, ntrials = 1,
                        PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                        FFT_PARAMS = FFT_PARAMS,
                        classiName = f"CNN_Classifier", path = path)

        clasificador.loadTrainingSignalPSD(filename = signalPSDName, path = path) #cargamos el PSD de mis datos de entrenamiento

    os.chdir(actualFolder)

    return clasificador

def clasificar(rawEEG, modelo, clasificador, anchoVentana = 5, bw = 2., order = 6, axis = 0):

    #### TO DO: Agregar los clasificadores que faltan. ###
    
    rawEEG = np.mean(rawEEG, axis = 0) #promediamos sobre los canales

    if modelo == "SVM":
        featureVector = clasificador.featuresExtraction(rawDATA = rawEEG, ventana = windows.hamming,
                        anchoVentana = anchoVentana, bw = bw, order = order, axis = axis, usePearson=True, applybank = False)

        comando = clasificador.getClassification(featureVector = featureVector)

    if modelo == "CNN":
        featureVector = clasificador.featuresExtraction(rawDATA = rawEEG, ventana = windows.hamming,
                        anchoVentana = anchoVentana, bw = bw, order = order, axis = axis, usePearson=True)
        comando = clasificador.getClassification(featureVector = featureVector)

    return comando


""" ######################################################
                        COMENZAMOS
######################################################"""

def main():

    """ ######################################################
    PASO 1: Cargamos datos generales de la sesión
    ######################################################"""

    placas = {"cyton": BoardIds.CYTON_BOARD, #IMPORTANTE: frecuencia muestreo 256Hz
              "ganglion": BoardIds.GANGLION_BOARD, #IMPORTANTE: frecuencia muestro 200Hz
              "synthetic": BoardIds.SYNTHETIC_BOARD}
    
    placa = placas["ganglion"] #7cd7
    electrodos = "pasivos"

    cantCanalesAUsar = 2 #Cantidad de canales a utilizar
    canalesAUsar = [1,2] #Seleccionamos canal uno y dos. NOTA: Si quisieramos elegir el canal 2 solamente debemos hacer [2,2] o [1,1] para elegir el canal 1

    cantidadTrials = 6 #cantidad de trials. Sirve para la sesión de entrenamiento.
    trialsAPromediar = 2
    contadorTrials = 0
    flagConTrials = True
    trials = cantidadTrials * trialsAPromediar
    trialDuration = 6 #secs #IMPORTANTE: trialDuration SIEMPRE debe ser MAYOR a stimuliDuration
    stimuliDuration = 4 #secs

    classifyData = True #True en caso de querer clasificar la señal de EEG
    fm = BoardShim.get_sampling_rate(placa)    
    window = stimuliDuration #segundos

    equipo = "mentalink"

    if equipo == "mentalink":
        frecStimulus = np.array([7, 85, 10])
        listaEstims = frecStimulus.copy().tolist()
        movements = [b'1', b'2', b'3',b'4'] #1 adelante, 2 izq, 3 derecha, 4 atrás

    if equipo == "neurorace":
        frecStimulus = np.array([11, 7, 9]) #11:adelante, 7:izquierda, 9:derecha, 13:atrás
        listaEstims = frecStimulus.copy().tolist()
        movements = [b'1', b'2', b'3',b'4']

    """ ##########################################################################################
    PASO 2: Cargamos datos necesarios para el clasificador y cargamos clasificador
    ##########################################################################################"""

    actualFolder = os.getcwd() #Folder base
    path = os.path.join(actualFolder,"models")

    modelo = "svm" #seleccionamos un modelo

    if modelo == "svm":
        #### Cargamos clasificador SVM ###
        modelName = "svm_waltertwo_linear" #Nombre archivo que contiene el modelo SVM
        signalPSDName = "svm_waltertwo_linear_signalPSD.txt"
        modeloClasificador = "SVM"

        PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = os.path.join(actualFolder,"models"))
        descarteInicial = int(fm*PRE_PROCES_PARAMS['ti']) #en segundos
        descarteFinal = int(window*fm)-int(fm*PRE_PROCES_PARAMS['tf']) #en segundos
        nsamples = int((window - PRE_PROCES_PARAMS['ti'] - PRE_PROCES_PARAMS['tf'])*fm)
        clasificador = cargarClasificador(modelo = modeloClasificador, modelName = modelName, signalPSDName = signalPSDName,
                                        frecStimulus = frecStimulus, nsamples = nsamples, path = path)
        
        #Cargamos parámetros. Usaremos algunos
        PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = os.path.join(actualFolder,"models"))

    if modelo == "cnn":

        ## Cargamos clasificador CNN ###
        modeloClasificador = "CNN"
        modelName = "cnntesting"
        modelFile = f"{modelName}.h5" #nombre del modelo
        signalPSDName = "cnntesting_signalPSD.txt"
        clasificador = cargarClasificador(modelo = modeloClasificador, modelName = modelName, signalPSDName = signalPSDName,
                                        frecStimulus = frecStimulus, nsamples = nsamples, path = path)
        
        #Cargamos parámetros. Usaremos algunos
        PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = os.path.join(actualFolder,"models"))

    descarteInicial = int(fm*PRE_PROCES_PARAMS['ti']) #en segundos
    descarteFinal = int(window*fm)-int(fm*PRE_PROCES_PARAMS['tf']) #en segundos
    anchoVentana = (window - PRE_PROCES_PARAMS['ti'] - PRE_PROCES_PARAMS['tf']) #fm * segundos

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
    time.sleep(1) #esperamos 2 segundos

    #### CONFIGURAMOS LA PLACA CYTON O GANGLION######
    """
    IMPORTANTE: No tocar estos parámetros.
    El string es:
    x (CHANNEL, POWER_DOWN, GAIN_SET, INPUT_TYPE_SET, BIAS_SET, SRB2_SET, SRB1_SET) X

    Doc: https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
    Doc: https://docs.openbci.com/Ganglion/GanglionSDK/
    """

    if placa == BoardIds.GANGLION_BOARD.value:
        canalesAdesactivar = ["3","4"]
        for canal in canalesAdesactivar:
            board_shim.config_board(canal) #apagamos los canales 3 y 4
            time.sleep(1)

    if placa == BoardIds.CYTON_BOARD.value:
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

    """ ##########################################################################################
    PASO 4: Iniciamos la recepción de dataos desde la placa OpenBCI
    ##########################################################################################"""
    
    board_shim.start_stream(450000, args.streamer_params) #iniciamos OpenBCI. Ahora estamos recibiendo datos.
    time.sleep(2)
    
    """genero un objeto DataThread para extraer datos de la OpenBCI"""
    data_thread = DT(board_shim, args.board_id)
    #graph = Graph(board_shim)
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
    arduino = AC('COM10', trialDuration = trialDuration, stimONTime = stimuliDuration,
             timing = 100, ntrials = trials)

    time.sleep(1) 
    arduino.iniSesion() #Inicio sesión en el Arduino.
    time.sleep(1) 

    EEGTrialsAveraged = []

    try:
        #graph.start()
        while arduino.generalControl() == b"1":

            if classifyData and arduino.systemControl[1] == b"0": #se apagan los estímulos y chequeamos si estamos para clasificar la señal de eeg
                contadorTrials += 1
                currentData = data_thread.getData(stimuliDuration)
                EEGTrialsAveraged.append(currentData)
                if contadorTrials == trialsAPromediar:
                    rawEEG = np.asarray(EEGTrialsAveraged).mean(axis = 0)
                    rawEEG = rawEEG[canalesAUsar[0]-1:canalesAUsar[1], descarteInicial:descarteFinal]
                    rawEEG = rawEEG - rawEEG.mean(axis = 1, keepdims=True) #resto media la media a la señal
                    # print("tipo",type(arduino.estadoRobot),arduino.estadoRobot)
                    print(f'Obstaculos en: {arduino.estadoRobot}')
                    clasificador.obstacles = str(arduino.estadoRobot) #actalizamos tabla de obstáculos
                    if clasificador.obstacles == '0111': #sólo podemos mover el vehículo hacia atrás
                        print(f"Comando a enviar {movements[3]}")
                        arduino.systemControl[2] = movements[3]
                        esadoRobot = arduino.sendMessage(arduino.systemControl) #Enviamos mensaje a Arduino con el comando clasificado
                    else:
                        frecClasificada = clasificar(rawEEG, modeloClasificador, clasificador, anchoVentana = anchoVentana, bw = 2., order = 4, axis = 0)
                        print(f"Comando a enviar {movements[listaEstims.index(frecClasificada)]}. Frecuencia {frecClasificada}")
                        arduino.systemControl[2] = movements[listaEstims.index(frecClasificada)]
                        esadoRobot = arduino.sendMessage(arduino.systemControl) #Enviamos mensaje a Arduino con el comando clasificado
                    contadorTrials = 0
                    EEGTrialsAveraged = []
                classifyData = False

            elif classifyData == False and arduino.systemControl[1] == b"1": #Se encienden estímulos y se habilita para más adelante la clasificación
                arduino.systemControl[2] = b'0' #cargamos movimiento STOP
                classifyData = True
        
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
        arduino.close() #cierro comunicación serie para liberar puerto COM
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

        
    finally:
        # graph.keep_alive = False
        # graph.join()
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()
            
        arduino.close() #cierro comunicación serie para liberar puerto COM

if __name__ == "__main__":
        main()