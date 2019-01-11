import pyaudio as pa
import sys
import numpy as np
import pa_np_interface as pn
from bi_aural import probe_signal
from io import BytesIO

read_samples = 1024

# Create a probe signal
fc = 7000  # Hz
bw = 12000  # Hz
fs = 44.1e3  # Hz
duty_cycle = 0.5
T = 0.6  # second
lfm = probe_signal.LFM(duty_cycle, fc, bw, T, fs)

raw_out = np.array(lfm.signal)
bb = pn.ByteBuffer(raw_out)

def find_UA101():
    """Find device with UA-101 in string"""
    for i in range(p.get_device_count()):
        name = p.get_device_info_by_index(i)['name']
        if 'UA-101' in name:
            return i

chunck_size = int(read_samples * bb.num_out_channels * bb.num_bytes)
with BytesIO(bb.all_bytes) as f:
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt24,
                channels=bb.num_out_channels,
                rate=int(fs),
                output_device_index=find_UA101(),
                output=True)
    data = f.read(chunck_size)
    while data:
        stream.write(data)
        data = f.read(chunck_size)
    stream.stop_stream()
    stream.close()
    p.terminate()
