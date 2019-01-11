import numpy as np
import scipy.signal as sig

class Processor:
    """bandpass, match filter, hilbert"""
    def __init__(self, replica, f_bounds, fs, array_length=None):
        """ setup for signal processing
        Replica vector is not time reversed
        f_bounds for bandpass filter
        set array length may allow for some pre-calculation
        """
        self.mf = np.flipud(np.array(replica))
        self.f_bounds = f_bounds
        self.fs = fs  # Hz
        self._create_bandpass()
        self.processor = None  # precompute filter values
        if array_length is not None:
            NFFT = int(2 ** np.ceil(np.log2(array_length)))
            f1_FT = np.fft.fft(sp_er.bp, NFFT)
            f2_FT = np.fft.fft(sp_er.mf, NFFT)
            self.processor = f1_FT * f2_FT

    def __call__(self, signal):
        """Returned processed signal"""
        if self.processor is None:
            NFFT = int(2 ** np.ceil(np.log2(signal.shape[1])))
            f1_FT = np.fft.fft(self.bp, NFFT)
            f2_FT = np.fft.fft(self.mf, NFFT)
            processor = f1_FT * f2_FT
        else:
            processor = self.processor
            NFFT = self.processor.size

        result = []
        for chan in signal:
            chan_FT = np.fft.fft(chan, NFFT)
            chan_FT = chan_FT * processor
            chan_FT[0: NFFT // 2] = 0 + 0 * 1j
            ts = 2 * np.fft.ifft(chan_FT)
            # Crop off unecassary data
            filter_length = self.mf.size + self.bp.size
            result.append(ts[filter_length: -filter_length])
        return np.array(result)

    def _create_bandpass(self):
        """Create a bandpass FIR filter"""
        num_taps = int(2 ** 8)
        beta = 1.5 * np.pi
        self.bp = sig.firwin(num_taps, self.f_bounds, window=('kaiser', beta),
                             pass_zero=False, nyq=self.fs/2)

if __name__ == "__main__":
    from bi_aural import probe_signal
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
    mf = np.flipud(np.array(mf.signal))
    f_bounds = (fc - bw / 2, fc + bw / 2)
    procesor = Processor(mf, f_bounds, fs)

    signal = procesor.bp
    NFFT = int(2 ** np.log2(np.ceil(signal.size) + 3))
    FT = np.fft.fft(signal, NFFT)
    f = np.arange(NFFT) / NFFT * fs

    FT = 20 * np.log10(np.abs(FT) + np.spacing(1))
    FT -= np.max(FT)

    fig, ax = plt.subplots()
    ax.plot(f, FT)
    ax.set_ylim(-50, 3)

    plt.show(block=False)
