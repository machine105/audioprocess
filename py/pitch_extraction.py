import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wio
import matplotlib.pyplot as plt

# returns the fundamental frequency[Hz]
# data: numpy 1d array
# Fs:   sampling rate[Hz]
def pitch(data, Fs, outImage=False):
    if data.ndim != 1:
        raise TypeError('data must be 1d numpy array')
    # int array -> float array
    data = data.astype('float32')
    
    N = data.shape[0]
    # Normalized Autocorrelation(v)
    r0 = np.dot(data, data)
    shift = np.copy(data)
    v = np.zeros(N, dtype='float32')
    v[0] = 1.0
    for i in range(1, N):
        shift = np.roll(shift, N - 1)
        shift[N - 1] = 0.0
        v[i] = np.dot(data, shift) / r0

    # output graph of v
    if outImage:
        kaxis = np.arange(0.0, 1.0 * N / Fs, 1.0 / Fs)
        plt.title('autocorrelation')
        plt.xlabel('time delay[sec]')
        plt.plot(kaxis, v)
        plt.savefig('../data/img/autocorrelation.png')
    
    # detect peek
    peeks = sig.argrelmax(v)
    maxpeek = 0.0
    delay = 0.0
    for peek in peeks[0]:
        if maxpeek < v[peek]:
            maxpeek = v[peek]
            delay = float(peek) / Fs
    if delay == 0:
        delay = -1
    F0 = 1.0 / delay
    return F0

# returns the temporal change of the fundamental frequency
# data: 1d numpy array
# Fs:   sampling rate[Hz]
def track_pitch(data, Fs, N, shift):
    length = data.shape[0]
    ret = np.zeros((length - N) / shift) 
    t_axis = np.zeros((length - N) / shift)
    for i in range(ret.shape[0]):
        ret[i] = pitch(data[i * shift: i * shift + N], Fs)
        t_axis[i] = i * float(shift) / Fs
    return ret, t_axis

if __name__ == '__main__':
    Fs, data = wio.read('../data/wav/test0.wav')
    data = data[:, 0] * 0.5 + data[:, 1] * 0.5
    ch, time = track_pitch(data, Fs, 1000, 500)
    plt.figure()
    plt.yscale('log')
    plt.plot(time, ch)
    plt.title('pitch')
    plt.xlabel('time')
    plt.ylabel('pitch[Hz]')
    plt.show()
