"""
Created on Fri Jul 30 12:26:03 2021
@author: Lucas
"""

import serial
import time
import keyboard

class Arduino:
    def __init__(self, port, trialDuration = 6, stimONTime = 4,
                 timeSleep = 0.1):
        
        self.dev = serial.Serial(port, baudrate=19200)
        
        self.timer = 0
        self.trialDuration =   int(trialDuration/timeSleep) #segundos
        self.stimONTime = int(stimONTime/timeSleep) #segundos
        self.stimOFFTime = int((trialDuration - stimONTime)/timeSleep)
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
        
        time.sleep(2)
        

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
        print("Sesión iniciada")
        
    def endSesion(self):
        self.sessionStatus = b"0"; #sesión en marcha
        self.leftStim = b"0"; #apago estimulos
        self.rightStim = b"0"; #estímulo derecho OFF
        self.backStim = b"0"; #estímulo hacia atras ON
        self.upperforwardStim = b"0"; #estímulo derecho ON
        
        self.stimuliState = [self.sessionStatus,
                        self.leftStim,
                        self.rightStim,
                        self.backStim,
                        self.upperforwardStim]
        
        self.sendStimuliState()
        print("Sesión Finlaziada")
        
        
    def trialControl(self):
        self.timer += 1
        if self.timer == self.stimONTime: #mandamos nuevo mensaje cuando comienza un trial
            self.stimuliState[1] = b"0" #apagamos estímulos
            self.sendStimuliState()
                
        if self.timer == self.trialDuration: 
            self.stimuliState[1] = b"1"
            self.sendStimuliState()
            print("Inicio nuevo trial")
            print("")
            
            self.trial += 1 #incrementamos un trial
            self.timer = 0 #reiniciamos timer
            
        return self.trial
    

def main():
    initialTime = time.time()#/1000
    
    sleepTime = 0.1
    ard = Arduino('COM3', timeSleep = sleepTime)
    
    
    trials = 2 #número de trials a ejecutar
    
    actualTrial = 1
    passTrial = actualTrial
    
    ard.iniSesion()
    print(f"Trial número {actualTrial}")
    
    while trials > 0:
        actualTrial = ard.trialControl()
        if actualTrial != passTrial:
            passTrial = actualTrial
            trials -= 1
            print(f"Trial número {actualTrial}")
        time.sleep(sleepTime)
        
    ard.endSesion()    
    ard.close()
    
    stopTime = time.time()#/1000
    
    print(f"Tiempo transcurrido en segundos: {stopTime - initialTime}")

if __name__ == "__main__":
    main()
    
# initialTime = time.time()#/1000

# while True:
#     if(time.time()-initialTime >= 1):
#         print("segundo")
#         initialTime = time.time()#/1000
    
    
    