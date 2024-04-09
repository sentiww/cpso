import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

SAMPLING_FREQUENCY = 360
FILE_PATH = '../data/ekg100.txt'
DIRECTORY = "images"


def read_ecg100_signal():
    ecg100 = pd.read_csv(FILE_PATH, names=['values'])
    ecg100['time'] = ecg100['values'].index / SAMPLING_FREQUENCY
    return ecg100


def calculate_fft(ecg100):
    average_signal_value = np.mean(ecg100['values'])
    ecg100['values'] = ecg100['values'] - average_signal_value
    fft = np.fft.fft(ecg100['values'])
    freq_spectral = np.fft.fftfreq(ecg100['values'].size, 1 / SAMPLING_FREQUENCY)
    amplitude_spectrum = np.abs(fft)[:len(freq_spectral) // 2] // 2
    freqs_positive = freq_spectral[:len(freq_spectral) // 2]
    return amplitude_spectrum, freqs_positive, fft


def plot(values, arguments, x_lim, y_lim):
    plt.figure(figsize=(14, 8))
    plt.legend()
    plt.grid(True)
    plt.plot(arguments, values)
    plt.xlim(300, 302)
    plt.ylim(-1, 1.2)


def compare_ifft_on_plot(original_signal, ifft):
    plt.figure(figsize=(10, 4))
    plt.plot(original_signal['values'][:1000], label='Original Signal')
    plt.plot(ifft[:1000], label='IFFT Signal', linestyle='--')
    plt.legend()
    plt.xlabel('Próbki')
    plt.ylabel('Sygnal [mV]')
    plt.title('Comparison of Original and IFFT Signals')
    plt.grid(True)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    plt.savefig(os.path.join(DIRECTORY, "orignal_ifft_comparison"))


def main():
    ecg100 = read_ecg100_signal()
    amplitude_spectrum, freqs_positive, fft = calculate_fft(ecg100)
    inverse_fft = np.fft.ifft(fft)
    difference = ecg100['values'] - inverse_fft.real

    plt.figure(figsize=(14, 8))
    plt.plot(ecg100['time'], ecg100['values'])
    plt.title('Sygnał ekg100.txt')
    plt.xlabel('Czas [s]')
    plt.ylabel('Sygnal [mV]')
    plt.xlim(300, 302)
    plt.ylim(-1, 1.2)
    plt.grid(True)
    plt.legend()
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    plt.savefig(os.path.join(DIRECTORY, "ekg100_signal_plot"))

    plt.figure(figsize=(14, 8))
    plt.plot(freqs_positive, amplitude_spectrum)
    plt.title('Widmo amplitudowe sygnału ekg100')
    plt.ylabel('Amplituda')
    plt.xlabel('Częstotliwość [Hz]')
    plt.grid(True)
    plt.legend()
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    plt.savefig(os.path.join(DIRECTORY, "ekg100_amplitude_spectrum"))

    compare_ifft_on_plot(ecg100, inverse_fft)

    plt.figure(figsize=(14, 8))
    plt.plot(ecg100['time'], difference, label='Różnica między sygnałem oryginalnym a odtworzonym')
    plt.title('Różnica między sygnałem oryginalnym a odtworzonym z IFFT')
    plt.xlabel('Czas [s]')
    plt.ylabel('Różnica [mV]')
    plt.grid(True)
    plt.xlim(300, 302)
    plt.legend()
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    plt.savefig(os.path.join(DIRECTORY, "original_ifft_diffrence"))


if __name__ == '__main__':
    main()
