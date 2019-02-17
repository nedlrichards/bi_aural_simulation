import numpy as np
from math import pi
from scipy.signal import firwin, convolve, resample
from scipy.interpolate import interp1d

class NoteProcessing:
    """Signal processing to work with short time tonal signals"""
    def __init__(self, fc, f_bb, record_threshold, fs):
        """Basic setup"""
        self.fc = fc
        # baseband information
        self.bw = 500
        self.f_bb = f_bb  # final frequency of signal after baseband
        # fudge factor on either side of filter for non-perfect edges
        self.edge_factor = 1.5
        # set generous upper limit if possible, but keep filter symetric
        self.f_lower = self.fc - self.bw * self.edge_factor
        self.f_upper = self.fc + self.bw * self.edge_factor
        # minimum sampling frequency after downsampling
        self.fcutoff = (self.f_upper - self.fc) * self.edge_factor + f_bb
        self.fcutoff *= 2
        # set event trigger threshold
        self.record_threshold = record_threshold
        self.fs = fs
        self.blocksize = 4096  # record and write size
        self.numblocks = 16  # number of blocks that make up a record
        self.record_length = self.blocksize * self.numblocks
        self.NFFT = 2 * self.blocksize
        # common fft frequency axis
        self.faxis = np.arange(self.NFFT) / self.NFFT
        self.faxis *= self.fs
        # index of frequencies of interest
        self.fci = np.zeros(self.faxis.size, dtype=np.bool_)
        self.fci = np.bitwise_and(self.faxis > fc - self.bw,
                                  self.faxis < fc + self.bw)
        # construnct band pass filter used before baseband operation
        numtaps = 1024 + 1
        edges = [self.f_lower, self.f_upper]
        self.bp_filt = firwin(numtaps, edges, window='blackmanharris',
                              pass_zero=False, fs=fs)
        self.bp_FT = np.fft.fft(self.bp_filt, n=self.record_length)
        # Add a Hilbert xform to bandpass filter to make signal analytic
        self.bp_FT[1:self.record_length // 2] *= 2
        self.bp_FT[self.record_length // 2: ] *= 0
        # integer downsampling, keep every Nth sample
        self.decimation = int(np.ceil(self.fs / self.fcutoff))
        # keep private time axis for recording before downsampling
        self._taxis = np.arange(self.record_length, dtype=np.float_)
        self._taxis /= fs
        # keep public time axis for recording after downsampling
        self.taxis = self._taxis[: : self.decimation]
        # beamforming specifications
        self.beam_upsample = 4

    def baseband(self, data_in):
        """baseband data"""
        # bandpass filter input data
        in_data_FT = np.fft.fft(data_in, axis=0)
        in_data_FT *= self.bp_FT[:, None]
        recorded_data_bb = np.fft.ifft(in_data_FT, axis=0)
        # demodulate signal with complex carrier
        f_shift = self.fc - self.f_bb
        phase_bb = np.exp(-2j * pi * f_shift * self._taxis)[:, None]
        recorded_data_bb *= phase_bb
        # downsample data before returning
        data_final = recorded_data_bb[: : self.decimation, :]
        return data_final

    def beamform(self, baseband_data, relative_delays):
        """Construct a time domain interpolator for data, delay and sum"""
        data_up = resample(baseband_data[:, 1:],
                           self.taxis.size * self.beam_upsample)
        dt = (self.taxis[-1] - self.taxis[0]) / (self.taxis.size - 1)
        taxis_up = np.arange(data_up.shape[0]) * dt / self.beam_upsample

        # intilize beam data to first channel
        beam_out = np.tile(baseband_data[:, 0], (relative_delays.shape[0], 1))
        beam_out = beam_out.T

        # frequency of demodulation
        f_shift = self.fc - self.f_bb

        for delays, channel in zip(relative_delays[:, 1: ].T, data_up.T):
            data_ier = interp1d(taxis_up, channel, kind='cubic',
                                fill_value=0, bounds_error=False)
            beam_times = self.taxis[:, None] + delays[None, :]
            # remove baseband phase shift from each channel
            delayed_chans = data_ier(beam_times)
            delayed_chans *= np.exp(2j * pi * f_shift * delays[None, :])
            # add each channel with delay to the reference channel
            beam_out += delayed_chans
        # running average of beamformer power
        beam_out = np.abs(beam_out) ** 2
        N = int(np.ceil(1 / (dt * self.bw)))
        # running average, SO # 13728392
        cs = np.cumsum(beam_out, axis=0)
        beam_out = (cs[N:, :] - cs[:-N, :]) / N
        return beam_out
