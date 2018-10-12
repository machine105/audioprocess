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

if __name__ == '__main__':
    src = source(240, 44100, 50)
    data = np.real(np.fft.ifft(src))
    wio.write('../data/wav/source.wav', 44100, data)
