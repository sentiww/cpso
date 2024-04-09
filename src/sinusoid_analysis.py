import numpy as np
import matplotlib.pyplot as plt
import os

LENGTH = 65536
SAMPLING_FREQUENCIES = [256, 512, 1024, 2048]
DIRECTORY = "images"


class Signal:
    def __init__(self, values, sampling_freq, plot_name):
        self.values = values
        self.sampling_freq = sampling_freq
        self.plot_name = plot_name


def generate_signal(frequencies, sampling_frequency, length):
    time = np.arange(length) / sampling_frequency
    signal = sum(a * np.sin(2 * np.pi * freq * time) for a, freq in frequencies)
    if len(frequencies) > 1:
        return Signal(signal, sampling_frequency, f'amplitude_spectrum_{sampling_frequency}_50_60')
    return Signal(signal, sampling_frequency, f'amplitude_spectrum_{sampling_frequency}_50')


def calculate_fft(signal, sampling_frequency):
    fft_result = np.fft.fft(signal)
    amplitude_spectrum = np.abs(fft_result)
    fft_frequency = np.fft.fftfreq(len(signal), 1 / sampling_frequency)
    return amplitude_spectrum, fft_frequency, fft_result


def calculate_ifft(fft_signal):
    return np.fft.ifft(fft_signal)


def plot_fft(amplitude_spectrum, fft_freqs, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs[:len(fft_freqs) // 2], amplitude_spectrum[:len(fft_freqs) // 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Spectrum')
    plt.grid(True)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    plt.savefig(os.path.join(DIRECTORY, filename))


def compare_ifft(original_signal, ifft, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(original_signal[:1000], label='Original Signal')
    plt.plot(ifft[:1000], label='IFFT Signal', linestyle='--')
    plt.legend()
    plt.title('Comparison of Original and IFFT Signals')
    plt.grid(True)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    plt.savefig(os.path.join(DIRECTORY, filename))


def main():
    signals: list[Signal] = [
        generate_signal(config, sampling_freq, LENGTH)
        for sampling_freq in SAMPLING_FREQUENCIES
        for config in [
            [(10, 50)],
            [(10, 50), (30, 60)]
        ]
    ]

    for sig in signals:
        amplitude_spectrum, fft_freq, fft = calculate_fft(sig.values, sig.sampling_freq)
        plot_fft(amplitude_spectrum, fft_freq, sig.plot_name)
        ifft = calculate_ifft(fft)
        compare_ifft(sig.values, ifft, sig.plot_name + "_ifft")


if __name__ == '__main__':
    main()
