import numpy as np
import pyaudio as pa
import pa_np_interface
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import scipy.signal as sig
from io import BytesIO

T = 3  # second

# input data
read_samples = 1024 * 4
chunck_size = int(read_samples * 6)

# play and record callback
record = []
def callback(in_data, frame_count, time_info, status):
    record.append(in_data)
    put_data = None
    return (None, pa.paContinue)

# setup usb device
def find_UA101():
    """Find device with UA-101 in string"""
    for i in range(p.get_device_count()):
        name = p.get_device_info_by_index(i)['name']
        if 'UA-101' in name:
            return i


p = pa.PyAudio()
stream = p.open(format=pa.paInt24,
	    channels=2,
	    rate=int(44.1e3),
	    input=True,
	    input_device_index=find_UA101(),
	    stream_callback=callback)


stream.start_stream()
start_time = time.time()
elapsed_time = 0
try:
    while elapsed_time < T:
        time.sleep(0.1)
        elapsed_time = time.time() - start_time
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

dc = pa_np_interface.Dechunker()
recorded = dc.buf_to_np(b''.join(record))
fig, ax = plt.subplots()
t_axis = np.arange(recorded.shape[0]) / 44.1e3
for ch, l in zip(recorded.T, ['ch 1: right', 'ch 2: left']):
    norm_channel = ch / np.max(np.abs(ch))
    ax.plot(t_axis * 1e3, norm_channel, label=l)

ax.set_xlabel('time, ms')
ax.set_ylabel('amplitude')
ax.grid()
ax.legend()

plt.show(block=False)
