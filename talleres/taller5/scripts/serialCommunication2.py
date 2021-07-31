# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:26:03 2021

@author: Lucas
"""

import serial
import time
import keyboard

class Arduino:
    def __init__(self, port):
        self.dev = serial.Serial(port, baudrate=19200)
        time.sleep(2)

    def query(self, message):
        self.dev.write(message)#.encode('ascii'))
        line = self.dev.readline().decode('ascii').strip()
        return line
    
    def close(self):
        self.dev.close()
    
ard = Arduino('COM3')

sessionStatus = b"1"; #sesión en marcha
leftStim = b"1"; #estímulo izquierdo ON
rightStim = b"0"; #estímulo izquierdo OFF
backStim = b"1"; #estímulo hacia atras ON
upperforwardStim = b"1"; #estímulo derecho ON
mensaje = [sessionStatus,
           leftStim,rightStim,
           backStim,upperforwardStim]

data = []
for byte in mensaje:
    data.append(ard.query(byte))
    
print(data)

sleepTime = 0.1 #segundos
trialDuration = 6 #segundos
data = []
timer = 0
trialDuration = int(trialDuration/sleepTime)

trials = 1; #espero a que corran los trials
        
while trials > 0:
    timer +=1
    time.sleep(sleepTime)
    if timer == trialDuration: #mandamos nuevo mensaje cuando comienza un trial
        timer = 0
        data = []
        for byte in mensaje:
            data.append(ard.query(byte))
        trials = trials - 1
    
print(data)
    
sessionStatus = b"0"; #sesión en marcha
leftStim = b"1"; #estímulo izquierdo ON
rightStim = b"0"; #estímulo izquierdo OFF
backStim = b"1"; #estímulo hacia atras ON
upperforwardStim = b"1"; #estímulo derecho ON
mensaje = [sessionStatus,
           leftStim,rightStim,
           backStim,upperforwardStim]

data = []
for byte in mensaje:
    data.append(ard.query(byte))
    
ard.close()
    

# data = []
# for _ in range(10):
#     data.append(ard.query(b'2'))

#print(data)

