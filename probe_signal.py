import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import wavfile

"""Create pulse signals and basic related information"""
class Probe:
    """Base class"""
    def __init__(self, fc, bw, period, fs):
        """Save basic information"""
        self.fc = fc
        self.bw = bw
        self.fs = fs
        self.period = period
        num_samples = period * fs
        self.time = np.arange(num_samples) / fs
        self.signal = np.zeros(self.time.shape)

    def FT(self):
        """Compute the signal's FT"""
        NFFT = int(2 ** np.log2(np.ceil(self.signal.size) + 1))
        FT = np.fft.fft(self.signal, NFFT)
        f = np.arange(NFFT) / NFFT * self.fs
        return (FT, f)

    def auto_corr(self):
        """Compute signal's auto-correlation"""
        corr = sig.fftconvolve(self.signal, np.flipud(self.signal), 'same')
        t_corr = np.arange(corr.size) / fs
        t_corr -= np.mean(t_corr)
        return (corr, t_corr)

class LFM:
    """A LFM chirp"""
    def __init__(self, duty_cycle, fc, bw, period, fs):
        """Duty cycle is length pulse is active"""
        self.pulse = Probe(fc, bw, period, fs)
        self.pulse.signal = self.lfm_chirp(duty_cycle)

    @property
    def signal(self):
        """convience"""
        return self.pulse.signal

    @property
    def time(self):
        """convience"""
        return self.pulse.time

    def lfm_chirp(self, duty_cycle):
        """create an lfm chirp to specs"""
        f1 = self.pulse.fc - self.pulse.bw / 2
        f2 = self.pulse.fc + self.pulse.bw / 2
        k = (f2 - f1) / duty_cycle
        t = np.array(self.time)
        t_i = t <= duty_cycle
        chirp = np.zeros(self.signal.shape)
        chirp[t_i] = np.sin(2 * np.pi * (f1 * t[t_i] +
                                    k / 2 * t[t_i] ** 2))
        # window?
        window = sig.kaiser(sum(t_i), 1.5 * np.pi)
        chirp[t_i] *= window

        return chirp

    def FT(self):
        """Inherit pulse method"""
        return self.pulse.FT()

    def auto_corr(self):
        """Inherit pulse method"""
        return self.pulse.auto_corr()

    def write_wave(self, file_name):
        """Composition method"""
        self.pulse.write_wave(file_name)

if __name__ == "__main__":
    fc = 7000  # Hz
    bw = 12000  # Hz
    fs = 44.1e3  # Hz
    duty_cycle = 0.5
    T = 0.6  # second
    lfm = LFM(duty_cycle, fc, bw, T, fs)
    FT, freq = lfm.FT()
    a_corr, t_corr = lfm.auto_corr()

    fig, ax = plt.subplots()
    ax.plot(lfm.time, lfm.signal)
    ax.set_xlabel('time, s')
    ax.set_ylabel('amplitude')
    ax.set_title('probe signal time series')

    fig, ax = plt.subplots()
    db_FT = 20 * np.log10(np.abs(FT))
    db_FT -= np.max(db_FT)
    ax.plot(freq, db_FT)
    ax.set_xlabel('frequency, Hz')
    ax.set_ylabel('Magnitude, dB')
    ax.set_title('probe signal fourier transform')

    fig, ax = plt.subplots()
    ax.plot(1e3 * t_corr, a_corr)
    ax.set_xlabel('time, ms')
    ax.set_ylabel('amplitude')
    ax.set_title('autocorrelation time series')
    ax.set_xlim(-1, 1)

    plt.show(block=False)
