import pyaudio as pa
import numpy as np

class UA101:
    """Buffer array and information related to the EDIROL UA101"""
    def __init__(self):
        """basic type inventory"""
        self.num_bytes = 3
        self.num_out_channels = 1
        self.num_in_channels = 2

    @property
    def device_ID(self):
        """Find pyaudio ID number. Returns None if not found"""
        #Find device with UA-101 in string
        for i in range(p.get_device_count()):
            name = p.get_device_info_by_index(i)['name']
            if 'UA-101' in name:
                return i

    def array_to_bytes(self, in_array):
        """Create a byte buffer of int24 type"""
        in_range = 2 ** (8 * self.num_bytes - 1)
        # Assume array has range of -1 to 1
        e = np.round(self.in_array * in_range).astype(np.dtype('<i4'))
        a = np.zeros((cast.size, self.num_bytes), np.dtype('<i1'))
        for i in reversed(range(self.num_bytes)):
            a[:, i] = e.view(dtype='<i1')[i + 1::4]
        # Copy array to a byte array
        out_bytes = a.tobytes('C')
        return out_bytes

    def bytes_to_array(self, bytes_buffer):
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
        result = e.astype(np.float32) * int_to_float
        # copy to take care of number of channels
        chans = []
        for i in range(self.num_in_channels):
            chans.append(result[i:: self.num_in_channels])
        result = np.vstack(chans).T
        return result


if __name__=='__main__':
    import probe_signal
    import matplotlib.pyplot as plt
    fc = 7000  # Hz
    bw = 12000  # Hz
    fs = 44.1e3  # Hz
    duty_cycle = 0.5
    T = 0.6  # second
    lfm = probe_signal.LFM(duty_cycle, fc, bw, T, fs)

    raw_out = np.array(lfm.signal)
    bb = UA101Buffer(raw_out)
    dc = Dechunker()

    # Have I created an inverse pair?
    all_bytes = bb.all_bytes
    full_loop = dc.buf_to_np(all_bytes)

    fig, ax = plt.subplots()
    comp_range = 40
    ax.plot(raw_out[: comp_range], 'b')
    ax.plot(full_loop[: comp_range, 0], 'g--')
    plt.show(block=False)
