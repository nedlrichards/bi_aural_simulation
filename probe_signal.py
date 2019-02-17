import numpy as np
import scipy.signal as sig

"""Create pulse signals and basic related information"""
class Probe:
    """Base class"""
    def __init__(self, num_samples, fs):
        """Save basic information"""
        self.num_samples = num_samples
        self.fs = fs
        self.period = num_samples * fs
        self.time = np.arange(num_samples) / fs
        self.signal = None

    def FT(self):
        """Compute the signal's FT"""
        NFFT = int(2 ** np.ceil(np.log2(self.signal.size) + 4))
        FT = np.fft.rfft(self.signal, NFFT)
        f = np.arange(NFFT // 2 + 1) / NFFT * self.fs
        return (FT, f)

    def auto_corr(self):
        """Compute signal's auto-correlation"""
        corr = sig.fftconvolve(self.signal, np.flipud(self.signal), 'same')
        t_corr = np.arange(corr.size) / fs
        t_corr -= np.mean(t_corr)
        return (corr, t_corr)

class LFM:
    """A LFM chirp"""
    def __init__(self, duty_cycle, fc, bw, fs):
        """Duty cycle is length pulse is active"""
        num_samples = int(np.ceil(duty_cycle * fs))
        # make even
        if num_samples % 2:
            num_samples += 1
        period = num_samples / fs
        self.fc = fc
        self.bw = bw
        self.pulse = Probe(num_samples, fs)
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
        f1 = self.fc - self.bw / 2
        f2 = self.fc + self.bw / 2
        k = (f2 - f1) / duty_cycle
        t = self.time
        chirp = np.zeros(self.pulse.num_samples)
        chirp = np.sin(2 * np.pi * (f1 * t + k / 2 * t ** 2))
        # window?
        window = sig.kaiser(self.pulse.num_samples, 2.0 * np.pi)
        chirp *= window

        return chirp

    def FT(self):
        """Inherit pulse method"""
        return self.pulse.FT()

    def auto_corr(self):
        """Inherit pulse method"""
        return self.pulse.auto_corr()

class NarrowBand:
    """A narrow banded pulse"""
    def __init__(self, fc, bw, fs):
        """Duty cycle is length pulse is active"""
        period = 1 / bw  # basic fourier transform uncertainty
        num_samples = int(np.ceil(period * fs))
        # make even
        if num_samples % 2:
            num_samples += 1
        self.fc = fc
        self.bw = bw
        self.fs = fs
        self.pulse = Probe(num_samples, fs)
        signal = self.narrow_band()
        self.pulse.signal = signal

    @property
    def signal(self):
        """convience"""
        return self.pulse.signal

    @property
    def time(self):
        """convience"""
        return self.pulse.time

    def narrow_band(self):
        """create an narrow banded pulse to specs"""
        xmitt = np.sin(2 * np.pi * self.fc * self.time)
        # window is unknown, assuming a pretty narrow mainlobe
        window = sig.kaiser(self.time.size, 1.0 * np.pi)
        xmitt *= window
        return xmitt

    def FT(self):
        """Inherit pulse method"""
        return self.pulse.FT()

    def auto_corr(self):
        """Inherit pulse method"""
        return self.pulse.auto_corr()
