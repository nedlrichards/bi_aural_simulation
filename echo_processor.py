import numpy as np
import scipy.signal as sig

class Processor:
    """bandpass, match filter, hilbert"""
    def __init__(self, replica, f_bounds, fs, NFFT, num_chan):
        """ setup for signal processing
        Replica vector is not time reversed
        f_bounds for bandpass filter
        set array length may allow for some pre-calculation
        """
        self.f_bounds = f_bounds
        self.fs = fs  # Hz
        self.bp = self._create_bandpass()
        self.mf = np.array(replica)[:: -1]
        # evens just make life simpler
        if NFFT % 2 == 1:
            NFFT += 1
        self.NFFT = NFFT
        self.num_chan = num_chan
        # center time axis to account for filter shift
        self.filter_length = self.bp.size // 2 + self.mf.size - 1
        if self.filter_length > NFFT:
            err = 'Filter ({} sample) is longer than NFFT ({} samples'.format(
                    self.filter_length, self.NFFT)
            raise(ValueError(err))
        self.taxis = np.arange(NFFT) / fs
        self.taxis -= self.filter_length / fs
        self._hilbFT = np.zeros((num_chan, NFFT), dtype=np.complex_)
        f1_FT = np.fft.rfft(self.bp, NFFT)
        f2_FT = np.fft.rfft(self.mf, NFFT)
        self.filt_FT = f1_FT * f2_FT

    def __call__(self, signal):
        """Returned processed signal"""
        # check that NFFT is appropriate
        signal = np.array(signal, ndmin=2)
        if signal.shape[-1] > self.NFFT:
            raise(ValueError('Signal is longer than NFFT, increase NFFT'))
        result = []
        for i, chan in enumerate(signal):
            chan_FT = np.fft.rfft(chan, self.NFFT)
            chan_FT = chan_FT * self.filt_FT
            self._hilbFT[i, : self.NFFT // 2 + 1] = 2 * chan_FT
        ts = 2 * np.fft.ifft(self._hilbFT)
        return np.squeeze(ts)

    def _create_bandpass(self):
        """Create a bandpass FIR filter"""
        num_taps = int(2 ** 8) + 1
        beta = 1.5 * np.pi
        bp = sig.firwin(num_taps, self.f_bounds, window=('kaiser', beta),
                        pass_zero=False, nyq=self.fs/2)
        return bp

if __name__ == "__main__":
    import probe_signal
    import matplotlib.pyplot as plt
    # Create a probe signal
    fc = 7000  # Hz
    bw = 12000  # Hz
    fs = int(44.1e3)  # Hz
    duty_cycle = 0.5
    T = 0.7  # second
    num_cycles = 2  # seconds
    rough_time = num_cycles * T + 0.1
    # Specify recording parameters
    lfm = probe_signal.LFM(duty_cycle, fc, bw, T, fs)
    # replica vector

    mf = probe_signal.LFM(duty_cycle, fc, bw, duty_cycle, fs)
    f_bounds = (fc - bw / 2, fc + bw / 2)

    procesor = Processor(mf.signal, f_bounds, fs, lfm.time.size, 1)

    sig_out = procesor(mf.signal)

    fig, ax = plt.subplots()
    ax.plot(procesor.taxis, np.real(sig_out))
    ax.plot(procesor.taxis, np.imag(sig_out))
    #ax.set_ylim(-50, 3)

    plt.show(block=False)
