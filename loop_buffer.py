from io import BytesIO
import probe_signal, pa_np_interface
import numpy as np

class LoopBuffer:
    """A buffer that loops a specified number of times"""
    def __init__(self, signal):
        self.bb = pa_np_interface.ByteBuffer(np.array(signal))
        self.num_out_channels = self.bb.num_out_channels
        self.num_bytes = self.bb.num_bytes
        self.current_loop = 0
        self.out_buffer = None
        self.chunck_size = None
        self.num_it = None

    def __call__(self, num_it, chunck_size):
        """Setup the buffer for a read cycle"""
        self.num_it = num_it
        self.current_loop = 0
        self.chunck_size = chunck_size
        self.out_buffer = BytesIO(self.bb.all_bytes)

    def read(self):
        """Return chuck size, restart buffer if necassary"""
        out = self.out_buffer.read(self.chunck_size)
        if len(out) < self.chunck_size:
            self.current_loop += 1
            if self.current_loop < self.num_it:
                self.out_buffer = BytesIO(self.bb.all_bytes)
                new_data = self.out_buffer.read(self.chunck_size - len(out))
                out = out + new_data
        return out
