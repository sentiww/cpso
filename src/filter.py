#!/usr/bin/python3
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', dest='filepath', help='Filepath to filter')

    args = parser.parse_args()

    return args


def load_file(filepath: str) -> list[(float, float)]:
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()
    
    values: list[float] = []
    for line in lines:
        tempValues = line.split()
        values.append((float(tempValues[0]), float(tempValues[1])))

    return values


def get_frequency_amplitude_response(values: list[(float, float)]) -> (np.ndarray, np.ndarray):
    timestamps, data = zip(*values)

    time_diff = np.diff(timestamps)
    frequency = 1 / np.mean(time_diff)

    fft_result = np.fft.fft(data)
    amplitude_spectrum = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(timestamps), d=time_diff[0])

    return (frequencies[:len(frequencies)//2], amplitude_spectrum[:len(frequencies)//2])


def plot_frequency_amplitude_response(values: list[(float, float)]):
    data = get_frequency_amplitude_response(values)

    plt.figure(figsize=(10, 6))
    plt.plot(data[0], data[1])
    plt.title('Frequency Amplitude Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def get_low_pass_cheby(sampling_frequency: float, cutoff_frequency: float):
    rp = 0.1
    rs = 40
    nyquist_freq = 0.5 * sampling_frequency
    norm_cutoff_freq = cutoff_frequency / nyquist_freq
    btype = 'low'

    N = 6

    numerator_coeffs, denominator_coeffs = signal.cheby1(N, rp, norm_cutoff_freq, btype, analog=False)

    return numerator_coeffs, denominator_coeffs


def get_high_pass_cheby(sampling_frequency: float, cutoff_frequency: float):
    rp = 0.1
    rs = 40
    nyquist_freq = 0.5 * sampling_frequency
    norm_cutoff_freq = cutoff_frequency / nyquist_freq
    btype = 'high'

    N = 6

    numerator_coeffs, denominator_coeffs = signal.cheby1(N, rp, norm_cutoff_freq, btype, analog=False)

    return numerator_coeffs, denominator_coeffs


def plot_low_pass_filtered_data(values: list[(float, float)]):
    time, value = zip(*values)
    time = np.array(time)
    value = np.array(value)
    
    sampling_frequency = 1000
    low_cutoff_frequency = 60
    low_numerator_coeffs, low_denominator_coeffs = get_low_pass_cheby(sampling_frequency, low_cutoff_frequency)

    filtered_values = signal.filtfilt(low_numerator_coeffs, low_denominator_coeffs, value)

    plt.figure(figsize=(14, 10))

    # Original data plot
    plt.subplot(3, 2, 1)
    plt.plot(time, value, 'b-')
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Filtered data plot
    plt.subplot(3, 2, 2)
    plt.plot(time, filtered_values, 'r-')
    plt.title('Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Frequency response plot
    w, h = signal.freqz(low_numerator_coeffs, low_denominator_coeffs, fs=sampling_frequency)
    plt.subplot(3, 2, 3)
    plt.plot(0.5 * sampling_frequency * w / np.pi, np.abs(h), 'g')
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)

    # Spectrum of original signal
    plt.subplot(3, 2, 4)
    original_fft = np.fft.fft(value)
    original_freq = np.fft.fftfreq(len(value), 1 / sampling_frequency)
    plt.plot(original_freq, np.abs(original_fft), 'b')
    plt.title('Signal Spectrum - Original')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Spectrum of filtered signal
    plt.subplot(3, 2, 5)
    filtered_fft = np.fft.fft(filtered_values)
    filtered_freq = np.fft.fftfreq(len(filtered_values), 1 / sampling_frequency)
    plt.plot(filtered_freq, np.abs(filtered_fft), 'r')
    plt.title('Signal Spectrum - Filtered')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_high_pass_filtered_data(values: list[(float, float)]):
    time, value = zip(*values)
    time = np.array(time)
    value = np.array(value)
    
    sampling_frequency = 1000
    high_cutoff_frequency = 5
    high_numerator_coeffs, high_denominator_coeffs = get_high_pass_cheby(sampling_frequency, high_cutoff_frequency)

    filtered_values = signal.filtfilt(high_numerator_coeffs, high_denominator_coeffs, value)

    plt.figure(figsize=(14, 10))

    # Original data plot
    plt.subplot(3, 2, 1)
    plt.plot(time, value, 'b-')
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Filtered data plot
    plt.subplot(3, 2, 2)
    plt.plot(time, filtered_values, 'r-')
    plt.title('Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Frequency response plot
    w, h = signal.freqz(high_numerator_coeffs, high_denominator_coeffs, fs=sampling_frequency)
    plt.subplot(3, 2, 3)
    plt.plot(0.5 * sampling_frequency * w / np.pi, np.abs(h), 'g')
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)

    # Spectrum of original signal
    plt.subplot(3, 2, 4)
    original_fft = np.fft.fft(value)
    original_freq = np.fft.fftfreq(len(value), 1 / sampling_frequency)
    plt.plot(original_freq, np.abs(original_fft), 'b')
    plt.title('Signal Spectrum - Original')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Spectrum of filtered signal
    plt.subplot(3, 2, 5)
    filtered_fft = np.fft.fft(filtered_values)
    filtered_freq = np.fft.fftfreq(len(filtered_values), 1 / sampling_frequency)
    plt.plot(filtered_freq, np.abs(filtered_fft), 'r')
    plt.title('Signal Spectrum - Filtered')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_band_pass_filtered_data(values: list[(float, float)]):
    time, value = zip(*values)
    time = np.array(time)
    value = np.array(value)
    
    sampling_frequency = 1000
    low_cutoff_frequency = 60
    high_cutoff_frequency = 5
    low_numerator_coeffs, low_denominator_coeffs = get_low_pass_cheby(sampling_frequency, low_cutoff_frequency)
    high_numerator_coeffs, high_denominator_coeffs = get_high_pass_cheby(sampling_frequency, high_cutoff_frequency)

    low_filtered_values = signal.filtfilt(low_numerator_coeffs, low_denominator_coeffs, value)
    band_filtered_values = signal.filtfilt(high_numerator_coeffs, high_denominator_coeffs, low_filtered_values)

    plt.figure(figsize=(14, 10))

    # Original data plot
    plt.subplot(2, 2, 1)
    plt.plot(time, value, 'b-')
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Filtered data plot
    plt.subplot(2, 2, 2)
    plt.plot(time, band_filtered_values, 'r-')
    plt.title('Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Spectrum of original signal
    plt.subplot(2, 2, 3)
    original_fft = np.fft.fft(value)
    original_freq = np.fft.fftfreq(len(value), 1 / sampling_frequency)
    plt.plot(original_freq, np.abs(original_fft), 'b')
    plt.title('Signal Spectrum - Original')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Spectrum of filtered signal
    plt.subplot(2, 2, 4)
    filtered_fft = np.fft.fft(band_filtered_values)
    filtered_freq = np.fft.fftfreq(len(band_filtered_values), 1 / sampling_frequency)
    plt.plot(filtered_freq, np.abs(filtered_fft), 'r')
    plt.title('Signal Spectrum - Filtered')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


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
    plot_low_pass_filtered_data(entries)

    # 3
    plot_high_pass_filtered_data(entries)

    # 4
    plot_band_pass_filtered_data(entries)