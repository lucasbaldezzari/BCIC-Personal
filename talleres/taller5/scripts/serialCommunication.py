# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:02:25 2021

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
    
ard = Arduino('COM3')

for _ in range(10):
    print(ard.query(b'1'))
    time.sleep(0.05)
    print(ard.query(b'0'))
    time.sleep(0.05)

data = []
for _ in range(10):
    data.append(ard.query(b'2'))

print(data)