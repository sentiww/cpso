#!/usr/bin/python3
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', dest='filepath', help='Filepath to filter')
    return parser.parse_args()


def load_file(filepath: str) -> list[(float, float)]:
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    values: list[float] = []
    for line in lines:
        tempValues = line.split()
        values.append((float(tempValues[0]), float(tempValues[1])))

    return values


def plot_frequency_amplitude_response(values: list[(float, float)]):
    timestamps, data = zip(*values)
    freqs, spectrum = calculate_spectrum(data, 360)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, spectrum)
    plt.xlim(-1, 75)
    plt.title('Frequency Amplitude Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('images', "task4_filtered_frequency_amplitude_response"))
    plt.show()


def get_pass_cheby(sampling_frequency: float, cutoff_frequency: int or [int, int], btype: str):
    rp = 0.1
    numerator_coeffs, denominator_coeffs = signal.cheby1(6, rp, cutoff_frequency, btype, analog=False, fs=sampling_frequency)

    return numerator_coeffs, denominator_coeffs


def calculate_filtered_values(values, sampling_frequency, cutoff_frequency: int or [int, int], btype: str):
    numerator_coeffs, denominator_coeffs = get_pass_cheby(sampling_frequency, cutoff_frequency, btype)
    return signal.filtfilt(numerator_coeffs, denominator_coeffs, values)


def calculate_spectrum(values, sampling_frequency):
    # values -= values.mean()
    spectrum = np.abs(np.fft.rfft(values)) / (len(values) // 2)
    frequencies = np.fft.rfftfreq(len(values), 1 / sampling_frequency)
    return frequencies, spectrum


def plot_pass_filtered_data(values: list[(float, float)], cutoff_frequency: float, btype: str):
    time, value = zip(*values)
    time = np.array(time)
    value = np.array(value)

    sampling_frequency = 1000
    numerator_coeffs, denominator_coeffs = get_pass_cheby(sampling_frequency, cutoff_frequency, btype)
    filtered_values = signal.filtfilt(numerator_coeffs, denominator_coeffs, value)

    freqs, spectrum = calculate_spectrum(filtered_values, sampling_frequency)

    plt.figure(figsize=(14, 8))
    plt.plot(freqs, spectrum, 'r')
    plt.title('Signal Spectrum - Filtered')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.xlim(-1, 80)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    w, h = signal.freqz(numerator_coeffs, denominator_coeffs, fs=sampling_frequency)
    h = np.abs(h)
    plt.plot(w, h, 'g')
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(time, filtered_values, 'r-')
    plt.title('Filtered Data')
    plt.xlabel('Time [s]')
    plt.ylabel('Value [mV]')
    plt.xlim(0, 2)
    plt.grid(True)
    plt.legend()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_band_pass_filtered_data(values: list[(float, float)]):
    time, value = zip(*values)
    time = np.array(time)
    value = np.array(value)
    sampling_frequency = 1000

    high_numerator_coeffs, high_denominator_coeffs = get_pass_cheby(sampling_frequency, cutoff_frequency=5, btype='high')
    low_filtered_values = calculate_filtered_values(value, sampling_frequency, 60, 'low')

    band_through_low_high_filter = signal.filtfilt(high_numerator_coeffs, high_denominator_coeffs, low_filtered_values)
    band_filtered_values = calculate_filtered_values(value, sampling_frequency, [5, 60], 'bandpass')

    plt.figure(figsize=(15, 12))

    freqs_low_high, spectrum_log_high = calculate_spectrum(band_through_low_high_filter, sampling_frequency)
    plt.subplot(2, 2, 1)
    plt.plot(freqs_low_high, spectrum_log_high, 'b')
    plt.xlim(0, 200)
    plt.title('Signal Spectrum with high and low filter')
    plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time, band_through_low_high_filter, 'r-')
    plt.title('Filtered signal with high and low filter')
    plt.xlabel('Time [s]')
    plt.ylabel('Value [mV]')
    plt.xlim(0, 4)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    freqs_band, spectrum_band = calculate_spectrum(band_through_low_high_filter, sampling_frequency)
    plt.subplot(2, 2, 3)
    plt.plot(freqs_band, spectrum_band, 'b')
    plt.xlim(0, 200)
    plt.title('Signal Spectrum with band filter')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time, band_filtered_values, 'r-')
    plt.title('Filtered signal with band filter')
    plt.xlabel('Time [s]')
    plt.ylabel('Value [mV]')
    plt.xlim(0, 4)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join('images', "filtered_band_plots"))


'''
    Example of running this script:
    filter.py --filepath ekg_noise_converted.txt
'''
if __name__ == '__main__':
    args = get_args()

    entries = load_file(args.filepath)

    # 1
    plot_frequency_amplitude_response(entries)

    # 2
    plot_pass_filtered_data(entries, cutoff_frequency=60, btype='low')

    # 3
    plot_pass_filtered_data(entries, cutoff_frequency=5, btype='high')

    # 4
    plot_band_pass_filtered_data(entries)
