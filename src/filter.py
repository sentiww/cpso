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


def plot_frequency_amplitude_response(data: (np.ndarray, np.ndarray)):
    plt.figure(figsize=(10, 6))
    plt.plot(data[0], data[1])
    plt.title('Frequency Amplitude Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def get_low_pass_cheby(sampling_frequency: float, cutoff_frequency: float):
    rp = 0.1            # Passband ripple (dB)
    rs = 40             # Stopband attenuation (dB)
    nyquist_freq = 0.5 * sampling_frequency
    norm_cutoff_freq = cutoff_frequency / nyquist_freq
    btype = 'low'

    # Manually specify the filter order
    N = 6  # You may need to adjust this based on your requirements

    # Designing the Chebyshev type I filter
    numerator_coeffs, denominator_coeffs = signal.cheby1(N, rp, norm_cutoff_freq, btype, analog=False)

    return numerator_coeffs, denominator_coeffs


def get_high_pass_cheby(sampling_frequency: float, cutoff_frequency: float):
    rp = 0.1            # Passband ripple (dB)
    rs = 40             # Stopband attenuation (dB)
    nyquist_freq = 0.5 * sampling_frequency
    norm_cutoff_freq = cutoff_frequency / nyquist_freq
    btype = 'high'

    # Manually specify the filter order
    N = 6  # You may need to adjust this based on your requirements

    # Designing the Chebyshev type I filter
    numerator_coeffs, denominator_coeffs = signal.cheby1(N, rp, norm_cutoff_freq, btype, analog=False)

    return numerator_coeffs, denominator_coeffs


def filter(values: list[(float, float)]):
    time, value = zip(*values)
    time = np.array(time)
    value = np.array(value)
    
    sampling_frequency = 1000
    cutoff_frequency = 60
    low_numerator_coeffs, low_denominator_coeffs = get_low_pass_cheby(sampling_frequency, cutoff_frequency)

    filtered_values = signal.filtfilt(low_numerator_coeffs, low_denominator_coeffs, value)

    plt.figure(figsize=(14, 10))

    # Original data plot
    plt.subplot(3, 2, 1)
    plt.plot(time, value, 'b-', label='Original')
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Filtered data plot
    plt.subplot(3, 2, 2)
    plt.plot(time, filtered_values, 'r-', label='Filtered')
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

    high_numerator_coeffs, high_denominator_coeffs = get_high_pass_cheby(sampling_frequency, cutoff_frequency)

    high_filtered_values = signal.filtfilt(high_numerator_coeffs, high_denominator_coeffs, value)

    both_filtered_values = signal.filtfilt(high_numerator_coeffs, high_denominator_coeffs, filtered_values)

    # Filtered data plot
    plt.subplot(3, 2, 4)
    plt.plot(time, high_filtered_values, 'r-', label='Filtered')
    plt.title('Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    # Frequency response plot
    w, h = signal.freqz(high_numerator_coeffs, high_denominator_coeffs, fs=sampling_frequency)
    plt.subplot(3, 2, 5)
    plt.plot(0.5 * sampling_frequency * w / np.pi, np.abs(h), 'g')
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid(True)

    # Filtered data plot
    plt.subplot(3, 2, 6)
    plt.plot(time, both_filtered_values, 'r-', label='Filtered')
    plt.title('Filtered Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


'''
    Example of running this script:
    filter.py --filepath ekg_noise_converted.txt
'''
if __name__ == '__main__':
    args = get_args()
    
    entries = load_file(args.filepath)
    filter(entries)
    #data = get_frequency_amplitude_response(entries)
    #plot_frequency_amplitude_response(data)