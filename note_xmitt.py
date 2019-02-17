import numpy as np
import scipy.signal as sig

class NarrowBand:
    """A narrow banded pulse"""
    def __init__(self, fc, bw, fs):
        """Duty cycle is length pulse is active"""
        period = 1 / bw  # basic fourier transform uncertainty
        num_samples = int(np.ceil(period * fs))
        # make even
        if num_samples % 2:
            num_samples += 1
        self.num_samples = num_samples
        self.period = num_samples * fs
        self.time = np.arange(num_samples) / fs
        self.fc = fc
        self.bw = bw
        self.fs = fs
        self.signal = self.narrow_band()

    def narrow_band(self):
        """create an narrow banded pulse to specs"""
        xmitt = np.sin(2 * np.pi * self.fc * self.time)
        # window is unknown, assuming a pretty narrow mainlobe
        window = sig.kaiser(self.num_samples, 1.0 * np.pi)
        xmitt *= window
        return xmitt
