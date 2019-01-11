import pyaudio as pa
import numpy as np

class ByteBuffer:
    """Create a buffer array suitable for py audio callback from array"""
    def __init__(self, in_array):
        """basic type inventory"""
        self.num_bytes = 3
        self.np_dtype = in_array.dtype
        self.in_array = in_array
        self.num_out_channels = 2
        self._bb = self._array_to_buffer()

    @property
    def all_bytes(self):
        """return all bytes at once"""
        return self._bb

    def _array_to_buffer(self):
        """Create a byte buffer of correct type"""
        in_range = 2 ** (8 * self.num_bytes - 1)
        # Assume array has range of -1 to 1
        cast = np.round(self.in_array * in_range)
        e = np.array(cast, np.dtype('<i4'))
        a = np.zeros((cast.size, self.num_bytes), np.dtype('<i1'))
        for i in reversed(range(self.num_bytes)):
            a[:, i] = e.view(dtype='<i1')[i + 1::4]
        # copy to take care of number of channels
        chans = []
        for _ in range(self.num_out_channels):
            chans.append(a)
        a = np.hstack(chans)
        # Copy array to a byte array
        out_bytes = a.tobytes('C')
        return out_bytes

class Dechunker:
    """Take chuncks of a byte buffer and convert to a np array"""
    def __init__(self):
        """Standard values for UA-101 board"""
        self.num_bytes = 3
        self.num_out_channels = 2

    def buf_to_np(self, buf):
        """Convert a buffer of bytes to numpy array
        in: 2 channel int24
        out: float32 numpy array"""
        self.num_bytes = 3
        # input is signed, hence - 1 in exponent
        in_range = 2 ** (8 * self.num_bytes - 1)
        # Convert values to 0-1 range
        int_to_float = 1 / in_range
        a = np.ndarray(len(buf), np.dtype('<i1'), buf)
        e = np.zeros(int(len(buf) // self.num_bytes), np.dtype('<i4'))
        for i in range(self.num_bytes):
            # e is offset by 1, this makes LSB 0 (up-casting data type)
            e.view(dtype='<i1')[i + 1::4] = a.view(dtype='<i1')[i::3]
        result = np.array(e, dtype='float32') * int_to_float
        # XXX: This is not num_channel aware
        # copy to take care of number of channels
        chans = []
        for i in range(self.num_out_channels):
            chans.append(result[i:: self.num_out_channels])
        result = np.vstack(chans).T
        return result

if __name__=='__main__':
    from bi_aural import probe_signal
    import matplotlib.pyplot as plt
    fc = 7000  # Hz
    bw = 12000  # Hz
    fs = 44.1e3  # Hz
    duty_cycle = 0.5
    T = 0.6  # second
    lfm = probe_signal.LFM(duty_cycle, fc, bw, T, fs)

    raw_out = np.array(lfm.signal)
    bb = ByteBuffer(raw_out)
    dc = Dechunker()

    # Have I created an inverse pair?
    all_bytes = bb.all_bytes
    full_loop = dc.buf_to_np(all_bytes)

    fig, ax = plt.subplots()
    comp_range = 40
    ax.plot(raw_out[: comp_range], 'b')
    ax.plot(full_loop[: comp_range, 0], 'g--')
    plt.show(block=False)



