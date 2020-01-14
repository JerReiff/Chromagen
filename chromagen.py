# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:01:12 2020

@author: jreif
"""

import numpy as np
from scipy.fftpack import fft


"""
min_frequencies computes the minimum frequency for each MIDI pitch bin.
It is of length 129 to give the MIDI pitch 127 an upper bound
"""
min_frequencies = 2**((np.arange(0,129)-69.5)/12)*440


"""
Create a Short-Time Fourier Transform of the given data.

data -- data array for STFT
windowlen -- sample size for each FFT
rate -- sample rate of data
windowtype -- "Rectangular" or "Hann" (controls shape of window function. Hann is recommended.) (default "Rectangular")

returns -- Array of frequencies, array of times, Chi-value  (weight) from STFT
"""

def stft(data, windowlen, rate, windowtype = "Rectangular"):
    if(windowtype == "Rectangular"):#This will manipulate the data at each slice to give different weights to values
        w = 1
    if(windowtype == "Hann"): #Hann is a window which goes to zero at the ends of the window, putting higher weight on the data
        u = np.linspace(-.5, .5, windowlen)#in the center of the window
        w = (1+np.cos(np.pi*u))/2
    hop = windowlen//4 #hop is the number of samples to move forward before performing the next FFT
    M = np.arange(0, (len(data)-windowlen)//hop-1, dtype=int) #M as an array makes it easier to create T_arr below, rather than making M an int.
    K_arr = np.arange(0, windowlen//2, dtype = int)
    Chi = np.empty((len(M), len(K_arr)), dtype=complex)
    for m in M:
        x = data[m*hop:windowlen+m*hop]
        chi_m = (fft(w*x))[:windowlen//2] #two notes here. First, we multiply by the window function before using fft.
                                        #second, for efficiency, we only calculate coefficients up to the nyquist frequency
        
        Chi[m,:] = chi_m
    F_arr = (K_arr*rate/windowlen)
    T_arr = (M*hop/rate)
    return abs(F_arr), abs(T_arr), abs(Chi)

"""
Create a chromagram of Fourier transform data

F_arr -- frequencies represented by the Fourier transform data
Chi -- Fourier transform data (can be STFT or FFT). Number of frequencies must be the length of F_arr.


"""
def chromagram(F_arr, Chi):
    return


