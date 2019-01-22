from io import BytesIO
import probe_signal, pa_np_interface
import numpy as np

class RepeatedTransmission:
    """A buffer that provides output a specified number of times"""
    def __init__(self, signal, chunck_size, num_loops):
        self.xmission = np.array(signal)
        self.ua = ua101_interface.UA101()
        self.chunck_size = int(chunck_size * self.num_out_channels * looper.num_bytes)
        self.num_loops = num_loops
        self.current_loop = None
        self.out_buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        """Reset buffer to initial state"""
        self.current_loop = 0
        self.out_buffer = self.ua.array_to_bytes(self.xmission)

    def __call__(self):
        """Return chuck size, restart buffer up to num_loops times"""
        out = self.out_buffer.read(self.chunck_size)
        if len(out) < self.chunck_size:
            self.current_loop += 1
            if self.current_loop < self.num_loops:
                self.out_buffer = self.ua.array_to_bytes(self.xmission)
                new_data = self.out_buffer.read(self.chunck_size - len(out))
                out = out + new_data
        return out
