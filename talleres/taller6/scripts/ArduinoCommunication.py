"""
Created on Fri Jul 30 12:26:03 2021
@author: Lucas

Clase Arduino.

Clase para comunicación entre Arduino y PC utilizando la libreria PYSerial
"""

import serial
import time
import keyboard

class ArduinoCommunication:
    """
    Clase para comunicación entre Arduino y PC utilizando la libreria PYSerial
    """
    def __init__(self, port, trialDuration = 6, stimONTime = 4,
                 timerFrecuency = 1000, timing = 1, useExternalTimer = False,
                 ntrials = None):
        
        self.dev = serial.Serial(port, baudrate=19200)
        
        self.trialDuration =   int((trialDuration*timerFrecuency)/timing) #segundos
        self.stimONTime = int((stimONTime*timerFrecuency)/timing) #segundos
        self.stimOFFTime = int((trialDuration - stimONTime))/timing*timerFrecuency
        self.stimStatus = "on"
        self.trial = 1
        self.trialsNumber = ntrials
        
 
        self.sessionStatus = b"1" #sesión en marcha
        self.stimuliStatus = b"0" #los estimulos empiezan apagados
        self.leftStim = b"1" #estímulo izquierdo ON
        self.rightStim = b"0" #estímulo derecho OFF
        self.backStim = b"1" #estímulo hacia atras ON
        self.forwardStim = b"1" #estímulo derecho ON
        self.stimuliState = [self.sessionStatus,
                             self.stimuliStatus,
                             self.leftStim,
                             self.rightStim,
                             self.backStim,
                             self.forwardStim]
        
        self.statusRequest = [b"2"]
        
        self.useExternalTimer = useExternalTimer
        self.timerEnable = 0
        self.timing = timing #en milisegundos
        self.timerFrecuency = timerFrecuency #1000Hz
        self.initialTime = 0 #tiempo inicial en milisegundos
        self.counter = 0
        self.timerInteFlag = 0 #flag para la interrupción del timer. DEBE ponerse a 0 para inicar una cuenta nueva.
        
        time.sleep(2) #esperamos 2 segundos para una correcta conexión
        
    def timer(self):
        """
        Función para emular un timer como el de un microcontrolador
        """
        if(self.timerInteFlag == 0 and
           time.time()*self.timerFrecuency - self.initialTime >= self.timing):
            self.initialTime = time.time()*self.timerFrecuency
            self.timerInteFlag = 1
            
    def iniTimer(self):
        """Iniciamos conteo del timer"""
        self.initialTime = time.time()*1000
        self.timerInteFlag = 0

    def query(self, message):
        """
        Enviamos un byte a arduinot y recibimos un byte desde arduino
        """
        self.dev.write(message)#.encode('ascii'))
        line = self.dev.readline().decode('ascii').strip()
        return line
    
    def sendMessage(self, message):
        """
        Función para enviar una lista de bytes con diferentes variables de estado
        hacia Arduino. Estas variables sirven para control de flujo del programa del
        Arduino.
        """
        incomingData = []
        for byte in message:
            incomingData.append(self.query(byte))
            
        return incomingData

    def close(self):
        """Cerramos comunicción serie"""
        self.dev.close()
        
    def iniSesion(self):
        """
        Se inicia sesión.
        """
        
        self.sessionStatus = b"1" #sesión en marcha
        self.stimuliStatus = b"1"; #empiezo a estimular
        self.leftStim = b"1" #estímulo izquierdo ON
        self.rightStim = b"1" #estímulo derecho OFF
        self.backStim = b"1" #estímulo hacia atras ON
        self.forwardStim = b"1" #estímulo derecho ON
        
        self.stimuliState = [self.sessionStatus,
                             self.stimuliStatus,
                             self.leftStim,
                             self.rightStim,
                             self.backStim,
                             self.forwardStim]
        
        self.sendMessage(self.stimuliState)
        self.iniTimer()
        print("Sesión iniciada")
        print("Trial inicial")

        
    def endSesion(self):
        """
        Se finaliza sesión.
        """
        
        self.sessionStatus = b"0" #sesión en marcha
        self.stimuliStatus = b"0"; #empiezo a estimular
        self.leftStim = b"0" #estímulo izquierdo ON
        self.rightStim = b"0" #estímulo derecho OFF
        self.backStim = b"0" #estímulo hacia atras ON
        self.forwardStim = b"0" #estímulo derecho ON
        
        self.stimuliState = [self.sessionStatus,
                             self.stimuliStatus,
                             self.leftStim,
                             self.rightStim,
                             self.backStim,
                             self.forwardStim]
        
        self.sendMessage(self.stimuliState)
        print("Sesión Finalizada")
        print(f"Trial final {self.trial}")
        
    def requestStatus(self):
        """
        Solicitamos estado del vehículo robótico al Arduino
        """
        return self.sendMessage(self.statusRequest) #retorno el mensaje recibido desde Arduino
        
        
    def trialControl(self):
        """
        Función que ayuda a controlar los estímulos en arduino.
            - La variable self.counter es utilizada como contador para sincronizar
            los momentos en que los estímulos están encendidos o opagados.
            - Es IMPORTANTE para un correcto funcionamiento que la variable self.counter
            se incremente en 1 de un tiempo adecuado. Para esto se debe tener en cuenta
            las variables self.stimONTime y self.trialDuration
        """

        self.counter += 1
        
        if self.counter == self.stimONTime: #mandamos nuevo mensaje cuando comienza un trial
        
            self.stimuliState[1] = b"0" #apagamos estímulos
            self.sendMessage(self.stimuliState)
            #estadoRobot = self.requestStatus()
              
        if self.counter == self.trialDuration: 
            
            self.stimuliState[1] = b"1"
            self.sendMessage(self.stimuliState)
            print(f"Fin trial {self.trial}")
            print("")
            self.trial += 1 #incrementamos un trial
            self.counter = 0 #reiniciamos timer
            
        return self.trial
    
    def generalControl(self):
        
        if self.stimuliState[0] == b"1" and not self.trialsNumber: #Para trials indefinidos
            
            if not self.useExternalTimer:
                self.timer()        
            if self.timerInteFlag: #timerInteFlag se pone en 1 a la cantidad de milisegundos de self.timing
                self.trialControl()
                self.timerInteFlag = 0 #reiniciamos flag de interrupción
                
        elif self.stimuliState[0] == b"1" and self.trial <= self.trialsNumber: #Para cantidad de trials definido
            
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

    """
    #creamos un objeto ArduinoCommunication para establecer una conexión
    #entre arduino y nuestra PC en el COM3, con un timing de 500ms y esperamos ejecutar
    #n trials.
    #Pasado estos n trials se finaliza la sesión.
    #En el caso de querer ejecutar Trials de manera indeterminada,
    #debe hacerse trials = None (default)
    """
    ard = ArduinoCommunication('COM3', timing = 500, ntrials = 12)

    # ard.iniTimer()
    ard.iniSesion()
    
    while ard.generalControl() == b"1":
        # if ard.trial-1 == 2:
        #     ard.endSesion()
        pass

    ard.endSesion()    
    ard.close()
    
    stopTime = time.time()#/1000
    
    print(f"Tiempo transcurrido en segundos: {stopTime - initialTime}")

if __name__ == "__main__":
    main()

    

    
    
    