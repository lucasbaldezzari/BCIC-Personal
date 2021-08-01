"""
Created on Fri Jul 30 12:26:03 2021
@author: Lucas
"""

import serial
import time

class Arduino:
    def __init__(self, port, trialDuration = 6, stimONTime = 4,
                 timeSleep = 0.1, timerFrecuency = 1000, timing = 1):
        
        self.dev = serial.Serial(port, baudrate=19200)
        
        self.trialDuration =   int((trialDuration*timerFrecuency)/timing) #segundos
        self.stimONTime = int((stimONTime*timerFrecuency)/timing) #segundos
        self.stimOFFTime = int((trialDuration - stimONTime))/timing*timerFrecuency
        self.stimStatus = "on"
        self.trial = 1
        
 
        self.sessionStatus = b"1" #sesión en marcha
        self.stimuliStatus = b"0" #los estimulos empiezan apagados
        self.leftStim = b"1" #estímulo izquierdo ON
        self.rightStim = b"0" #estímulo derecho OFF
        self.backStim = b"1" #estímulo hacia atras ON
        self.upperforwardStim = b"1" #estímulo derecho ON
        self.stimuliState = [self.sessionStatus,
                             self.stimuliStatus,
                             self.leftStim,
                             self.rightStim,
                             self.backStim,
                             self.upperforwardStim]
        
        self.timerEnable = 0
        self.timing = timing #in miliseconds
        self.timerFrecuency = timerFrecuency #1000Hz
        self.initialTime = 0 #get the time in miliseconds
        self.counter = 0
        self.timerInteFlag = 0 #flag for timer interruption. Must be set to 0 for a new count
        
        time.sleep(2)
        
    def timer(self):
        if(self.timerInteFlag == 0 and
           time.time()*self.timerFrecuency - self.initialTime >= self.timing):
            self.initialTime = time.time()*self.timerFrecuency#/1000
            self.timerInteFlag = 1
            
    def iniTimer(self):
        self.initialTime = time.time()*1000
        self.timerInteFlag = 0

    def query(self, message):
        self.dev.write(message)#.encode('ascii'))
        line = self.dev.readline().decode('ascii').strip()
        return line
    
    def sendStimuliState(self):
        
        incomingData = []
        for byte in self.stimuliState:
            incomingData.append(self.query(byte))
            
        return incomingData


    def close(self):
        self.dev.close()
        
    def iniSesion(self):
        
        self.sessionStatus = b"1" #sesión en marcha
        self.stimuliStatus = b"1"; #empiezo a estimular
        self.leftStim = b"1" #estímulo izquierdo ON
        self.rightStim = b"1" #estímulo derecho OFF
        self.backStim = b"1" #estímulo hacia atras ON
        self.upperforwardStim = b"1" #estímulo derecho ON
        
        self.stimuliState = [self.sessionStatus,
                             self.stimuliStatus,
                        self.leftStim,
                        self.rightStim,
                        self.backStim,
                        self.upperforwardStim]
        
        self.sendStimuliState()
        self.iniTimer()
        print("Sesión iniciada")

        
    def endSesion(self):
        
        self.sessionStatus = b"0" #sesión en marcha
        self.stimuliStatus = b"0"; #empiezo a estimular
        self.leftStim = b"0" #estímulo izquierdo ON
        self.rightStim = b"0" #estímulo derecho OFF
        self.backStim = b"0" #estímulo hacia atras ON
        self.upperforwardStim = b"0" #estímulo derecho ON
        
        self.stimuliState = [self.sessionStatus,
                             self.stimuliStatus,
                        self.leftStim,
                        self.rightStim,
                        self.backStim,
                        self.upperforwardStim]
        
        self.sendStimuliState()
        print("Sesión Finalizada")
        
    def trialControl(self):

        self.counter += 1
        
        if self.counter == self.stimONTime: #mandamos nuevo mensaje cuando comienza un trial
        
            self.stimuliState[1] = b"0" #apagamos estímulos
            self.sendStimuliState()
              
        if self.counter == self.trialDuration: 
            
            self.stimuliState[1] = b"1"
            self.sendStimuliState()
            print(f"Fin trial {self.trial}")
            print("")
            self.trial += 1 #incrementamos un trial
            self.counter = 0 #reiniciamos timer
            
        return self.trial
    
    def generalControl(self):
        
        self.timer()        
        if self.timerInteFlag: #timerInteFlag se pone en 1 a la cantidad de milisegundos de self.timing
            self.trialControl()
            self.timerInteFlag = 0 #reiniciamos flag de interrupción

    
def main():
    initialTime = time.time()#/1000

    ard = Arduino('COM3', timing = 500)

    trials = 5 #número de trials a ejecutar
    
    actualTrial = 1
    passTrial = actualTrial
    
    # ard.iniTimer()
    ard.iniSesion()
    print(f"Inicio trial número {actualTrial}")
    
    while trials > 0:
        ard.generalControl()
        actualTrial = ard.trial
        if actualTrial != passTrial:
            passTrial = actualTrial
            trials -= 1
            print(f"Inicio trial número {actualTrial}")
        
    ard.endSesion()    
    ard.close()
    
    stopTime = time.time()#/1000
    
    print(f"Tiempo transcurrido en segundos: {stopTime - initialTime}")

if __name__ == "__main__":
    main()

    

    
    
    