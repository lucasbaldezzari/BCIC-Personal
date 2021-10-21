"""
Created on Fri Jul 30 12:26:03 2021
@author: Lucas
Clase Arduino.
Clase para comunicación entre Arduino y PC utilizando la libreria PYSerial
        VERSIÓN: SCT-01-RevB (24/9/2021)
        Se agrega lista de movimientos en variable self.movements para enviar comandos a través del puerto serie
"""

import os
import serial
import time

class ArduinoCommunication:
    """Clase para comunicación entre Arduino y PC utilizando la libreria PYSerial.
        Constructor del objeto ArduinoCommunication
        
        Parametros
        ----------
        port: String
            Puerto serie por el cual nos conectaremos
        trialDuration: int
            Duración total de un trial [en segundos]
        stimONTime: int
            Duración total en que los estímulos están encendidos
        timerFrecuency: int
            Variable para "simular" la frecuencia [en Hz] de interrupción del timer
        timing: int
            Variable para temporizar interrupción - Por defecto es 1[ms]
        useExternalTimer: bool
            En el caso de querer que el timer funcione con una interrupción externa
        ntrials: int
            Cantidad de trials a ejecutar. Una vez pasados los ntrials, se deja de transmitir y recibir
            información hacia y desde el Arduino - Por defecto el valor es 1[trial] - Si se quisiera una
            ejecución por tiempo indeterminado se debe hacer ntrials = None
            
        Retorna
        -------
        Nada        
    """
    def __init__(self, port, trialDuration = 6, stimONTime = 4,
                 timerFrecuency = 1000, timing = 1, useExternalTimer = False,
                 ntrials = 1):
        
        self.dev = serial.Serial(port, baudrate=19200)
        
        self.trialDuration =   int((trialDuration*timerFrecuency)/timing) #segundos
        self.stimONTime = int((stimONTime*timerFrecuency)/timing) #segundos
        self.stimOFFTime = int((trialDuration - stimONTime))/timing*timerFrecuency
        self.stimStatus = "on"
        self.trial = 1
        self.trialsNumber = ntrials
        
        self.movements = [b'0',b'1',b'2',b'3',b'4',b'5'] #lista con los compandos
        """
        movements:
            b'0' = STOP (Neurorace) / ADELANTE (Mentalink)
            b'1' = ADELANTE (Neurorace) /  45° ADELANTE E IZQUIERDA (Mentalink)
            b'2' = LEFT (Neurorace) / IZQUIERDA (Mentalink)
            b'3' = ATRAS (Neurorace) / ATRAS (Mentalink)
            b'4' = DERECHA (Neuorace) / DERECHA (Mentalink)
            b'5' = 45° ADELANTE Y DERECHA (Mentalink)
            #El STOP de mentalink será self.moveOrder = b'63' (0b00111111)
        """

        self.sessionStatus = b"1" #sesión en marcha
        self.stimuliStatus = b"0" #los estimulos empiezan apagados
        self.moveOrder = self.movements[0] #EL robot empieza en STOP
        # self.moveOrder = b'63' #El STOP de mentalink será self.moveOrder = b'63' (0b00111111)
        """
        self.moveOrder
        - 0: STOP
        - 1: ADELANTE
        - 2: DERECHA
        - 3: ATRAS
        - 4: IZQUIERDA
        """
        self.systemControl = [self.sessionStatus,
                             self.stimuliStatus,
                             self.moveOrder]

        actualFolder = os.getcwd()
        self.stateFilePath = os.path.join(actualFolder,"visual stimulation")
        self.stateFile = "comunication.txt"
         
        self.useExternalTimer = useExternalTimer
        self.timerEnable = 0
        self.timing = timing #en milisegundos
        self.timerFrecuency = timerFrecuency #1000Hz
        self.initialTime = 0 #tiempo inicial en milisegundos
        self.counter = 0
        self.timerInteFlag = 0 #flag para la interrupción del timer. DEBE ponerse a 0 para inicar una cuenta nueva.
        
        time.sleep(2) #esperamos 2 segundos para una correcta conexión
        
    def timer(self):
        """Función para emular un timer como el de un microcontrolador"""
        if(self.timerInteFlag == 0 and
           time.time()*self.timerFrecuency - self.initialTime >= self.timing):
            self.initialTime = time.time()*self.timerFrecuency
            self.timerInteFlag = 1
            
    def iniTimer(self):
        """Iniciamos conteo del timer"""
        self.initialTime = time.time()*1000
        self.timerInteFlag = 0

    def query(self, byte):
        """Enviamos un byte a arduinot y recibimos un byte desde arduino
        
        Parametros
        ----------
        message (byte):
            Byte que se desa enviar por puerto serie.
        """
        self.dev.write(byte)#.encode('ascii')) #enviamos byte por el puerto serie
        respuesta = self.dev.readline().decode('ascii').strip() #recibimos una respuesta desde Arduino
        
        return respuesta
    
    def sendMessage(self, message):
        """Función para enviar una lista de bytes con diferentes variables de estado
        hacia Arduino. Estas variables sirven para control de flujo del programa del
        Arduino.
        """
        incomingData = []
        for byte in message:
            incomingData.append(self.query(byte))
            
        return incomingData[-1] #Retorno los últimos bytes recibidos

    def close(self):
        """Cerramos comunicción serie"""
        self.dev.close()
        
    def iniSesion(self):
        """Se inicia sesión."""
        
        self.sessionStatus = b"1" #sesión en marcha
        self.stimuliStatus = b"1" #encendemos estímulos
        self.moveOrder = b"0" #EL robot empieza en STOP
        
        self.systemControl = [self.sessionStatus,
                             self.stimuliStatus,
                             self.moveOrder]
        
        estadoRobot = self.sendMessage(self.systemControl)
        print("Estado inicial del ROBOT:", estadoRobot)
        
        #Actualizamos archivo de estados
        # estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2],
        #             int(estadoRobot[0]),
        #             int(estadoRobot[1]),
        #             int(estadoRobot[2])]

        estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2]]

        completeName = os.path.join(self.stateFilePath,self.stateFile)
        file = open(completeName, "w")
        for estado in estados:
            #file.write(str(estado) + "\n")
            file.write(str(estado))
        file.close()
        
        self.iniTimer()
        print("Sesión iniciada")
        print("Trial inicial")
        
    def endSesion(self):
        """Se finaliza sesión.
            Se envía información a Arduino para finalizar sesión. Se deben tomar acciones en el Arduino
            una vez recibido el mensaje.
        """
        
        self.sessionStatus = b"0" #sesión finalizada
        self.stimuliStatus = b"0" #finalizo estimulación
        self.moveOrder = b"0" #Paramos el rebot enviando un STOP
        self.systemControl = [self.sessionStatus,
                             self.stimuliStatus,
                             self.moveOrder]
        
        estadoRobot = self.sendMessage(self.systemControl)

        #Actualizamos archivo de estados
        # estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2],
        #             int(estadoRobot[0]),
        #             int(estadoRobot[1]),
        #             int(estadoRobot[2])]
        estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2]]

        completeName = os.path.join(self.stateFilePath,self.stateFile)
        file = open(completeName, "w")
        for estado in estados:
            #file.write(str(estado) + "\n")
            file.write(str(estado))
        file.close()

        print("Estado final del ROBOT:", estadoRobot)
        print("Sesión Finalizada")
        print(f"Trial final {self.trial - 1}")
        
    def trialControl(self):
        """Función que ayuda a controlar los estímulos en arduino.
            - La variable self.counter es utilizada como contador para sincronizar
            los momentos en que los estímulos están encendidos o opagados.
            - Es IMPORTANTE para un correcto funcionamiento que la variable self.counter
            se incremente en 1 de un tiempo adecuado. Para esto se debe tener en cuenta
            las variables self.stimONTime y self.trialDuration
        """

        self.counter += 1
        
        if self.counter == self.stimONTime: #mandamos nuevo mensaje cuando comienza un trial
        
            self.systemControl[1] = b"0" #apagamos estímulos
            estadoRobot = self.sendMessage(self.systemControl)
            print("Estado ROBOT:", estadoRobot)
            #Actualizamos archivo de estados
            # estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2],
            #             int(estadoRobot[0]),
            #             int(estadoRobot[1]),
            #             int(estadoRobot[2])]
            estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2]]

            completeName = os.path.join(self.stateFilePath,self.stateFile)
            file = open(completeName, "w")
            for estado in estados:
                #file.write(str(estado) + "\n")
                file.write(str(estado))
            file.close()
             
        if self.counter == self.trialDuration: 
            
            self.systemControl[1] = b"1"
            estadoRobot = self.sendMessage(self.systemControl)
            print("Estado ROBOT:", estadoRobot)

            #Actualizamos archivo de estados
            # estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2],
            #             int(estadoRobot[0]),
            #             int(estadoRobot[1]),
            #             int(estadoRobot[2])]

            estados = [str(self.systemControl[0])[2],str(self.systemControl[1])[2]]

            completeName = os.path.join(self.stateFilePath,self.stateFile)
            file = open(completeName, "w")
            for estado in estados:
                #file.write(str(estado) + "\n")
                file.write(str(estado))
            file.close()

            print(f"Fin trial {self.trial}")
            print("")
            self.trial += 1 #incrementamos un trial
            self.counter = 0 #reiniciamos timer
            
        return self.trial
    
    def generalControl(self):
        """Función para llevar a cabo un control general de los procesos entre PC y Arduino."""
        
        if self.systemControl[0] == b"1" and not self.trialsNumber: #Para trials indefinidos
            
            if not self.useExternalTimer:
                self.timer()        
                
            if self.timerInteFlag: #timerInteFlag se pone en 1 a la cantidad de milisegundos de self.timing
                self.trialControl()
                self.timerInteFlag = 0 #reiniciamos flag de interrupción
                
        elif self.systemControl[0] == b"1" and self.trial <= self.trialsNumber: #Para cantidad de trials definido
            
            if not self.useExternalTimer:    
                self.timer()   
                
            if self.timerInteFlag: #timerInteFlag se pone en 1 a la cantidad de milisegundos de self.timing
                self.trialControl()
                self.timerInteFlag = 0 #reiniciamos flag de interrupción
        
        else:
            self.endSesion()
                
        return self.sessionStatus
    
def main():
    
    initialTime = time.time()#/1000
    time.sleep(2)

    """
    #creamos un objeto ArduinoCommunication para establecer una conexión
    #entre arduino y nuestra PC en el COM3, con un timing de 500ms y esperamos ejecutar
    #n trials.
    #Pasado estos n trials se finaliza la sesión.
    #En el caso de querer ejecutar Trials de manera indeterminada,
    #debe hacerse trials = None (default)
    """
    ard = ArduinoCommunication('COM7', trialDuration = 8, stimONTime = 4,
                               timing = 100, ntrials = 1)
    time.sleep(1)
    ard.iniSesion()

    #Simulamos que enviamos el comando de movimiento número cuatro
    ard.systemControl[2] = ard.movements[3] #comando número 4 (b'1') [b'0',b'1',b'2',b'3',b'4',b'5']
    
    while ard.generalControl() == b"1":
        pass

    ard.endSesion()   
    ard.close() #cerramos comunicación serie y liberamos puerto COM
    
    stopTime = time.time()#/1000
    
    print(f"Tiempo transcurrido en segundos: {stopTime - initialTime}")

if __name__ == "__main__":
    main()