# -*- coding: utf-8 -*-
import scipy.io.wavfile as wio
import matplotlib.pyplot as plt
import numpy as np
import spectrum as spec
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
    def generate(filename, outname):
        # original voice
        Fs, voice = wio.read(filename)
        voice = voice.astype('float32') / 32767
        F0 = int(pitch.pitch(voice, Fs))

        # buzzer
        src = buzzer_source(60, Fs, 7.0).astype('float32')
        data = np.real(np.fft.ifft(src))
        wio.write('../data/wav/buzzer.wav', Fs, data)
        power = np.abs(src)

        # spectral envelope
        voice = spec.high_enphasis(voice)
        vowel = spec.envelope(voice, Fs)
    
        gen = power * np.exp(vowel)
        data = np.fft.ifft(gen)
        wio.write(outname, Fs, np.real(data).astype('float32'))

    generate('../data/wav/B_a.wav', '../data/wav/generated_a.wav')
    generate('../data/wav/A_i.wav', '../data/wav/generated_i.wav')
    generate('../data/wav/A_u.wav', '../data/wav/generated_u.wav')
    generate('../data/wav/A_e.wav', '../data/wav/generated_e.wav')
    generate('../data/wav/A_o.wav', '../data/wav/generated_o.wav')

