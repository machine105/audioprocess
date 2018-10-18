# -*- coding: utf-8 -*-
import scipy.io.wavfile as wio
import numpy as np

# returns pulse train in frequency domain
# it can be made audible by idft 
# F0:     fundamental frequency [Hz]
# Fs:     sampling rate (assumed even number)
# amp:    Amplitude?
# nPulse: number of pulse
def buzzer_source(F0, Fs, amp, nPulse=10):
    data = np.zeros(Fs)
    f = np.arange(F0, Fs / 2, F0)
    data[f] = amp
    # symmetry
    f = np.arange(Fs - F0, Fs / 2, -F0)
    data[f] = amp
    return data

def white_noise_source(Fs, amp):
    data = np.full(Fs, amp)
    return data

# returns synthesised sine wave in frequency domain
# freqs: list of frequency
# amps:  list of amplitude
# Fs:    sampling rate[Hz]
def sinewave_source(freqs, amps, Fs):
    data = np.zeros(Fs)
    for freq, amp in zip(freqs, amps):
        data[freq] = amp
        data[Fs - freq] = amp
    return data

if __name__ == '__main__':
    # buzzer
    src = buzzer_source(240, 44100, 50)
    data = np.real(np.fft.ifft(src))
    wio.write('../data/wav/source.wav', 44100, data)
    # white noise
    src = white_noise_source(44100, 50)
    data = np.real(np.fft.ifft(src))
    wio.write('../data/wav/noise.wav', 44100, data)
    # sine waves
    src = sinewave_source([440, 550], [500, 300], 44100)
    data = np.real(np.fft.ifft(src))
    wio.write('../data/wav/sine.wav', 44100, data)

