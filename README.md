# Chromagen
A module allowing for frequency analysis of .WAV files
Repo is a work in progress, and so far only STFT, spectrogram, and chromagram functionality have been implemented, which still relies on SciPy's FFT algorithm.
The module was written with the Anaconda Distribution, and currently requires installation of Scipy and Numpy, (and Matplotlib for the demo), though an effort may be made soon to remove Scipy from the list of dependencies.
An end goal is to be able to use machine learning classification algorithms on .WAV files such as k-means-clustering in order to group songs
