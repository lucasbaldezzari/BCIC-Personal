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

import threading

class GraphModule(threading.Thread):
    
    def __init__(self, board_shim):
        
        threading.Thread.__init__(self)
        
        pg.setConfigOption('background', '#F8F6F8') #F8F6F8
        pg.setConfigOption('foreground', '#1C1C1C')

        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)[6:14]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 1
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        
        self.keep_alive = True

    def _init_pens(self):
        self.pens = list()
        self.brushes = list()
        #colors = ['#A54E4E', '#A473B6', '#5B45A4', '#2079D2', '#32B798', '#2FA537', '#9DA52F', '#A57E2F', '#A53B2F']
        colors = ['#838283', '#7D4D8D', '#3E56A5', '#38705D', '#DCB302', '#F86335', '#DA422F', '#C6764E', '#C20664']
        
        for i in range(len(colors)):
            pen = pg.mkPen({'color': colors[i], 'width': 2})
            self.pens.append(pen)
            brush = pg.mkBrush(colors[i])
            self.brushes.append(brush)

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('EEG')
            self.plots.append(p)
            hist_pen = pg.mkPen((170, 57, 57, 255), width=1.)
            curve = p.plot(pen=self.pens[i % len(self.pens)])
            curve.setDownsampling(auto=True, method='mean', ds=3)
            self.curves.append(curve)

    def _init_psd(self):
        self.psd_plot = self.win.addPlot(row=0,col=1, rowspan=len(self.exg_channels)//2)
        self.psd_plot.showAxis('left', True)
        self.psd_plot.setMenuEnabled('left', True)
        self.psd_plot.setTitle('Espectro de Frecuencias')
        self.psd_plot.setLogMode(False, False)
        self.psd_curves = list()
        self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        for i in range(len(self.exg_channels)):
            psd_curve = self.psd_plot.plot(pen=self.pens[i % len(self.pens)])
            psd_curve.setDownsampling(auto=True, method='mean', ds=3)
            self.psd_curves.append(psd_curve)

    def _init_band_plot(self):
        self.band_plot = self.win.addPlot(row=len(self.exg_channels)//2, col=1, rowspan=len(self.exg_channels)//2)
        self.band_plot.showAxis('left', False)
        self.band_plot.setMenuEnabled('left', False)
        self.band_plot.showAxis('bottom', True)
        self.band_plot.showAxis('left', False)
        self.band_plot.setMenuEnabled('bottom', False)
        self.band_plot.setTitle('Bandas de potencia')
        y = [0, 0, 0, 0, 0]
        x = [1, 2, 3, 4, 5]
        self.band_bar = pg.BarGraphItem(x=x, height=y, width=0.8, pen=self.pens[0], brush=self.brushes[0])
        self.band_plot.addItem(self.band_bar)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        avg_bands = [0, 0, 0, 0, 0]
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 5.0, 38.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)
            
            self.curves[count].setData(data[channel].tolist())
            if data.shape[1] > self.psd_size:
                # plot psd
                psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2, self.sampling_rate,
                                    WindowFunctions.BLACKMAN_HARRIS.value)
                lim = min(70, len(psd_data[0]))
                self.psd_curves[count].setData(psd_data[1][0:lim].tolist(), psd_data[0][0:lim].tolist())
                # plot bands
                avg_bands[0] = avg_bands[0] + DataFilter.get_band_power(psd_data, 1.0, 4.0)
                avg_bands[1] = avg_bands[1] + DataFilter.get_band_power(psd_data, 4.0, 8.0)
                avg_bands[2] = avg_bands[2] + DataFilter.get_band_power(psd_data, 8.0, 13.0)
                avg_bands[3] = avg_bands[3] + DataFilter.get_band_power(psd_data, 13.0, 30.0)
                avg_bands[4] = avg_bands[4] + DataFilter.get_band_power(psd_data, 30.0, 50.0)

        avg_bands = [int(x * 100 / len(self.exg_channels)) for x in avg_bands]
        self.band_bar.setOpts(height=avg_bands)

        self.app.processEvents()
        
    def run(self):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot',size=(800, 600))

        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()
        
        # while True:
        #     if self.keep_alive == True:
        #         pass
        #     else:
        #         pg.QtGui.QApplication.quit()