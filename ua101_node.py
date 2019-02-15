import sounddevice as sd
import numpy as np
from math import pi
from queue import Queue
from scipy.signal import firwin, convolve, resample

class UA101:
    """Open an I/O interface with a UA101"""
    def __init__(self, fc, record_threshold):
        """Basic setup"""
        self.fc = fc
        self.record_threshold = record_threshold
        self.blocksize = 4096  # record and write size
        self.numblocks = 16
        self.bw = 500
        # sound device setup
        # find UA-101
        all_devs = sd.query_devices()
        ua_101_dev = [d for d in all_devs if 'UA-101' in d['name']]
        if len(ua_101_dev) == 1:
            self.deviceID = ua_101_dev[0]['name']
        else:
            raise(ValueError('UA-101 device not properly identified'))
        # remaining sd information
        self.channels = 2
        self.samplerate = int(ua_101_dev[0]['default_samplerate'])

        self.NFFT = 2 * self.blocksize
        self.faxis = np.arange(self.NFFT) / self.NFFT
        self.faxis *= self.samplerate

        # index of frequencies of interest
        self.fci = np.zeros(self.faxis.size, dtype=np.bool_)
        # used to go from an excedence to strongest center frequency
        self._fci_inverse = []
        for fc in self.fc:
            fci = np.bitwise_and(self.faxis > fc - self.bw,
                                 self.faxis < fc + self.bw)
            self._fci_inverse = np.ones(np.sum(fci)) * fc
            self.fci = np.bitwise_or(self.fci, fci)
        self._lastdata = Queue()
        self._all_data = Queue()
        self.last_center = None
        self.status = None

        # baseband information
        self.f_bb = 1.5 * self.bw
        self.fcut = 3 * self.bw
        numtaps = self.NFFT // 4 + 1
        edges = [self.f_bb - self.bw, self.f_bb + self.bw]
        self.bp_filt = firwin(numtaps, edges, window='blackmanharris',
                              pass_zero=False, fs=self.samplerate)

        self.bp_FT = np.fft.fft(self.bp_filt,
                                n=self.blocksize * self.numblocks)
        # Add a Hilbert xform to bandpass filter
        self.bp_FT[1:self.NFFT // 2] *= 2
        self.bp_FT[self.NFFT // 2: ] *= 0
        self.decimation = int(np.ceil(self.samplerate / self.fcut))
        self._taxis = np.arange(1. * self.blocksize * self.numblocks)
        self._taxis /= self.samplerate
        self.taxis = self._taxis[: : self.decimation]

    def record(self):
        """open a callback stream with UA101"""
        self.is_exceed = False
        kwargs = dict(samplerate=self.samplerate,
                       blocksize=self.blocksize,
                       device=self.deviceID,
                       channels=self.channels,
                       dtype='float32',
                       callback=self._callback)
        with sd.Stream(**kwargs) as s:
            while s.active:
                sd.sleep(int(100))

        # stack record in a single numpy array
        recorded_data = []
        while not self._all_data.qsize() == 0:
            recorded_data.append(self._all_data.get())
        recorded_data = np.concatenate(recorded_data)
        # empty out last data queue
        while not self._lastdata.qsize() == 0:
            self._lastdata.get()
        return recorded_data

    def baseband(self, data_in, is_inverse=False):
        """baseband data"""
        fbb = self.last_center - self.f_bb
        phase_bb = np.exp(-1j * 2 * pi * fbb * self._taxis)[:, None]
        if is_inverse:
            phase_bb = np.conj(phase_bb)
        in_data_FT = np.fft.fft(data_in * phase_bb, axis=0)
        if is_inverse:
            in_data_FT[1:self.NFFT // 2] /= 2
            # make data conjugate symetric
            in_data_FT[self.NFFT // 2: ] = \
                np.conj(in_data_FT[self.NFFT // 2])[::-1]
        else:
            in_data_FT *= self.bp_FT[:, None]
        recorded_data_bb = np.fft.ifft(in_data_FT, axis=0)
        return recorded_data_bb[: : self.decimation]

    def beamform(self, baseband_data, relative_delays):
        """Construct a time domain interpolator for data, delay and sum"""
        data_up = resample(baseband_data, self.taxis.size * 10)


    def _callback(self, indata, outdata, frames, time, status):
        """Check each block for excedence in frequencies of interest"""
        if status:
            self.status = status

        outdata.fill(0)
        curr_data = indata.copy()
        curr_data = curr_data.astype(np.float64)

        # trigger has been tripped, simply record untill reach number of blocks
        if self.is_exceed:
            self._all_data.put(curr_data)
            if self._all_data.qsize() >= self.numblocks - 1:
                raise(sd.CallbackStop)
            return

        # special case for first sample
        if self._lastdata.empty():
            self._lastdata.put(curr_data)
            return

        # construct array of this sample and last
        full_data = np.concatenate([self._lastdata.get(), curr_data])
        self._lastdata.put(curr_data)

        # process last 2 samples
        data_FT = np.fft.fft(full_data[:, 0])
        self.is_exceed = np.any(np.abs(data_FT[self.fci])
                                 > self.record_threshold)
        self.last_center = self._fci_inverse[np.argmax(np.abs(data_FT[self.fci]))]

        if self.is_exceed:
            self._all_data.put(full_data)
