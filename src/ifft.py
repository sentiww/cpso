import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

ecg100 = pd.read_csv(os.getcwd() + '/data/ekg100.txt', names=['values'])
sampling_frequency = 360

ecg100['time'] = ecg100['values'].index / sampling_frequency

plt.figure(figsize=(14, 8))
plt.plot(ecg100['time'], ecg100['values'])
plt.title('Widmo amplitudowe sygnałów')
plt.xlabel('Czas [s]')
plt.ylabel('Sygnal')
plt.xlim(0, 800)
plt.ylim(-3, 2.0)
plt.grid(True)
plt.legend()
plt.show()


fft = np.fft.fft(ecg100['values'])
freq_spectral = np.fft.fftfreq(ecg100['values'].size, d=1/sampling_frequency)
aplitude_spectral = np.abs(fft)[:len(fft)//2] / (len(fft) // 2)
freqs_positive = freq_spectral[:len(freq_spectral)//2]

plt.figure(figsize=(14, 8))
plt.plot(freqs_positive, aplitude_spectral)
plt.title('Widmo amplitudowe sygnału ekg100')
plt.xlabel('Częstotliwość [Hz]')
plt.xlim(0, sampling_frequency/2)
plt.ylim(0, 0.015)
plt.ylabel('Amplituda')
plt.grid(True)
plt.legend()
plt.show()

inverse_fft = np.fft.ifft(fft)
difference = ecg100['values'] - inverse_fft.real 

plt.figure(figsize=(14, 8))
plt.plot(ecg100['time'], difference, label='Różnica między sygnałem oryginalnym a odtworzonym')
plt.title('Różnica między sygnałem oryginalnym a odtworzonym z IDFT')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda różnicy')
plt.grid(True)
plt.legend()
plt.show()
