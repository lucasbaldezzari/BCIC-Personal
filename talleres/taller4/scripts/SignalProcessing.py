# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:40:18 2021

@author: Lucas Baldezzari

Signal Processing (SP) module
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:10:53 2021

@author: Lucas BALDEZZARI

Grpagication module using pyqtgraph.

Code apadted from https://github.com/brainflow-dev/brainflow/blob/master/python-package/examples/plot_real_time_min.py
"""

import argparse
import time
import logging
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

class signalProcessing:
    #something
    