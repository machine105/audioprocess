# -*- coding: utf-8 -*-
import scipy.io.wavfile as wio
import matplotlib.pyplot as plt
import numpy as np
import spectol as spec
import pitch_extraction as pitch

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
    # original voice
    Fs, voice = wio.read('../data/wav/A_a.wav')
    voice = voice.astype('float32') / 32767
    F0 = int(pitch.pitch(voice, Fs))

    # buzzer
    src = buzzer_source(240, Fs, 2.5).astype('float32')
    data = np.real(np.fft.ifft(src))
    wio.write('../data/wav/buzzer.wav', Fs, data)
    power = np.abs(src)
    log_power = np.array([np.log(p) if p>0 else 0 for p in power])
    plt.figure()
    plt.plot(log_power)

    # spectral envelope
    vowel = spec.envelope(voice, F0, Fs)
    plt.figure()
    plt.plot(vowel)
    
    gen = log_power + vowel
    plt.figure()
    plt.plot(np.exp(gen))
    data = np.fft.ifft(np.exp(gen))
    wio.write('../data/wav/generated_a.wav', Fs, 10*np.real(data).astype('float32'))

    plt.show()
