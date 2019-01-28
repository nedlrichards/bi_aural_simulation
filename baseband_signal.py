from scipy.signal import firwin, convolve
import numpy as np
from math import pi

class Basebander():
    """Baseband signal to have a spectral width of 3 bw"""
    def __init__(self, fc, bw, fs):
        """design baseband filter"""
        self.fc = fc
        self.bw = bw
        self.fs = fs
        self.fcut = (self.f_bb + 1.5 * bw) * 2
        self.bp_filt = firwin(2048 + 1, [self.fc - bw, self.fc + bw],
                              window='blackmanharris', pass_zero=False, fs=fs)

    def __call__(self, indata):
        """Baseband, bandpass and resample data"""
        NFFT = indata.shape[0]
        # bandpass filter data
        data_bp = convolve(indata, self.bp_filt[:, None], mode='valid')
        # baseband center frequency to baseband frequency
        taxis = (np.arange(data_bp.shape[0]) + self.bp_filt.size // 2) / self.fs
        taxis = taxis[:, None]
        bb_td = data_bp * np.exp(-1j * 2 * pi * (self.fc - self.f_bb) * taxis)
        # compute complex time series
        bb_FD = np.fft.fft(bb_td, axis=0)
        # create an analytic signal
        bb_FD[NFFT + 1:] = 0
        bb_td = np.fft.ifft(bb_FD, axis=0)
        # decimate sampling
        decimation = int(np.ceil(self.fs / self.fcut))
        # index notation stides array, saving only subset of samples
        return , bb_td[: : decimation, :]
