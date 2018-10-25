import numpy as np
import scipy.io.wavfile as wio
import matplotlib.pyplot as plt
import pitch_extraction as pitch

def power_spectol(data, Fs):
    # fundamental frequency
    spectol = np.fft.fft(data, Fs)
    power_spec = np.log(np.abs(spectol))
    return power_spec

def cepstrum(data, Fs):
    power_spec = power_spectol(data, Fs)
    cepstrum = np.fft.ifft(power_spec, Fs)
    return cepstrum
    
def envelope(data, Fs):
    c = cepstrum(data, Fs)
    c[45:] = 0
    env = np.fft.fft(c)
    return env

# +6 dB/oct
def high_enphasis(data):
    ret = np.copy(data)
    for k in range(data.shape[0] - 1):
        ret[k+1] = data[k+1] - 0.97 * data[k]
    return ret

if __name__ == '__main__':
    def draw_envelope(filename, **kwarg):
        Fs, data = wio.read(filename)
        env = envelope(data, Fs)
        plt.plot(np.abs(env[:Fs/2]), **kwarg)
        plt.xlabel('Frequency[Hz]')
    draw_envelope('../data/wav/A_a.wav', label='A')
    draw_envelope('../data/wav/B_a.wav', label='B')
    draw_envelope('../data/wav/C_a.wav', label='C')
    plt.legend()
    plt.savefig('../data/img/envelopes.png')
    plt.show()
