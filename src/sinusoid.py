import numpy as np
import matplotlib.pyplot as plt


def generate_signal(freq: int, length: int, sampling_freq: int): 
    time = np.arange(length) / sampling_freq
    signal = np.sin(2 * np.pi * freq * time)
    return generate_fft(signal, length, sampling_freq)


def generate_fft(signal: list, length, sampling_freq):
    signal_fft = np.fft.fft(signal)
    freq_spectral = np.fft.fftfreq(length, d=1./sampling_freq)
    aplitude_spectral = np.abs(signal_fft)
    return (freq_spectral, aplitude_spectral, sampling_freq)


def draw_signals(signals):
    plt.figure(figsize=(10, 5), dpi=100)

    for signal in signals:
        print(signal)
        freq_spectral, amplitude_spectral, sampling_freq = signal
        plt.plot(freq_spectral[:len(freq_spectral)//2], amplitude_spectral[:len(amplitude_spectral)//2], label=f'fs={sampling_freq}Hz')
    plt.title('Widmo amplitudowe sygnałów')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.legend()
    plt.show()

signals = [
    generate_signal(50, 65536, 65536),
    generate_signal(50, 65536, 1000),
    generate_signal(50, 65536, 5000),
    generate_signal(50, 65536, 10000),
    generate_signal(50, 65536, 50000),
    generate_signal(60, 65536, 65536),
    generate_signal(60, 65536, 1000),
    generate_signal(60, 65536, 5000),
    generate_signal(60, 65536, 10000),
    generate_signal(60, 65536, 50000),
]

