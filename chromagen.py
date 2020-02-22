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
Create a real-valued Short-Time Fourier Transform of the given data.

data -- data array for STFT
windowlen -- sample size for each FFT
rate -- sample rate of data
windowtype -- "Rectangular" or "Hann" (controls shape of window function. Hann is recommended.) (default "Rectangular")

returns -- Array of frequencies, array of times, Chi-value  (MAGNITUDE of coefficient) from STFT
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
    Chi = np.empty((len(K_arr), len(M)), dtype=complex)
    for m in M:
        x = data[m*hop:windowlen+m*hop]
        chi_m = (fft(w*x))[:windowlen//2] #two notes here. First, we multiply by the window function before using fft.
                                        #second, for efficiency, we only calculate coefficients up to the nyquist frequency
        
        Chi[:,m] = chi_m
    F_arr = (K_arr*rate/windowlen)
    T_arr = (M*hop/rate)
    return F_arr, T_arr, Chi #returns complex values in Chi
"""
Create a spectrogram of Fourier transform data

F_arr -- frequencies represented by the Fourier transform data
Chi -- Fourier transform data (can be STFT with Chi.shape[1]>1 or FFT with Chi.shape[1]==1). Number of frequencies must be the length of F_arr.

Returns mweights - size is (128,Chi.shape[1])
"""
def spectrogram(F_arr, Chi):
    pitch = np.full((F_arr.shape[0]),-1) #this will tell us to which MIDI pitch bin each frequency belongs
    i = 0
    j = 0
    while (F_arr[i] < min_frequencies[0] and i<F_arr.shape[0]):
        i+=1
    while (i<F_arr.shape[0] and j<128):#assign the pitches to MIDI notes
        if (F_arr[i]>=min_frequencies[j] and F_arr[i]<min_frequencies[j+1]):
            pitch[i] = j
            i+=1
        else:
            j=j+1
    mweights = np.zeros((128,Chi.shape[1]))
    for k in range(pitch.shape[0]): #sum over the relevant frequencies to get MIDI weights
        if (pitch[k]>-1):
            mweights[pitch[k],:] += np.abs(Chi[k,:]) #np.abs ensures we add the magnitude (Chi can be complex)
    return(mweights)

"""
Create a chromagram of STFT data, FFT data, or spectrogram data
If using with spectrogram data, specify spec_data=None

"""
def chromagram(F_arr=None,Chi=None, spec_data=None):
    
    if spec_data is None:
        if F_arr is None or Chi is None:
            return None
        return chromagram(spec_data = spectrogram(F_arr,Chi))
    else:
        cgm = np.zeros((12,spec_data.shape[1]))
        #maps midi pitch to chroma
        chroma = (np.arange(0,128)%12)
        for i in range(128):
            cgm[chroma[i],:] += spec_data[i,:]
        
        
        return np.roll(cgm,3, axis = 0) #MIDI pitches start at C, but this returns an array with note A = index 0




"""
Get the weight of each chroma over an entire chromagram
"""
def chromaweights(chromagram):
    #first normalize the chromagram at each time step
    #Then, to get each chroma's weight, take sum of that chroma at all time steps and divide by total number of time steps
    return np.sum(np.linalg.norm(chromagram,axis=0),axis=1)/np.shape(chromagram)[1]




