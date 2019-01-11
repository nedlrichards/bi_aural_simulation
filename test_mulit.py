import numpy as np
import pyaudio as pa
from bi_aural import probe_signal, pa_np_interface
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import scipy.signal as sig
from io import BytesIO
import queue

# Create a probe signal
fc = 7000  # Hz
bw = 12000  # Hz
fs = int(44.1e3)  # Hz
duty_cycle = 0.5
T = 0.7  # second
lfm = probe_signal.LFM(duty_cycle, fc, bw, T, fs)
# Specify recording parameters
duration = 1.5  # seconds

# input data
read_samples = int(2**12)
raw_out = np.array(lfm.signal)
bb = pa_np_interface.ByteBuffer(raw_out)
chunck_size = int(read_samples * bb.num_out_channels * bb.num_bytes)

# output method
dc = pa_np_interface.Dechunker()
q = queue.Queue()

# play and record callback
record = []
def callback(in_data, frame_count, time_info, status):
    q.put([in_data])
    #out_data = f.read(chunck_size)
    out_data = f.read(frame_count)
    return (out_data, pa.paContinue)

# setup usb device
def find_UA101(p):
    """Find device with UA-101 in string"""
    for i in range(p.get_device_count()):
        name = p.get_device_info_by_index(i)['name']
        if 'UA-101' in name:
            return i
test = []
with BytesIO(bb.all_bytes) as f:
    p = pa.PyAudio()
    chan_ID = find_UA101(p)
    stream = p.open(format=pa.paInt24,
                    channels=bb.num_out_channels,
                    rate=fs,
                    output=True,
                    input=True,
                    #frames_per_buffer=chunck_size,
                    output_device_index=chan_ID,
                    input_device_index=chan_ID,
                    stream_callback=callback)


    stream.start_stream()
    start_time = time.time()
    elapsed_time = 0
    while elapsed_time < duration:
        while not q.empty():
            test.append(dc.buf_to_np(b''.join(q.get())))

        time.sleep(0.1)
        elapsed_time = time.time() - start_time

    stream.stop_stream()
    stream.close()
    p.terminate()

#recorded = dc.buf_to_np(b''.join(record))
import sys; sys.exit("User break") # SCRIPT EXIT
t_axis = np.arange(recorded.shape[0]) / fs

# Match filter the output
mf = probe_signal.LFM(duty_cycle, fc, bw, duty_cycle, fs)
mf = np.flipud(np.array(mf.signal))
all_corr = []
for i in range(bb.num_out_channels):
    conv = sig.fftconvolve(recorded[:, i], mf, 'valid')
    h_conv = sig.hilbert(conv)
    all_corr.append(h_conv)
all_corr = np.array(all_corr)

t_corr = np.arange(all_corr.shape[1]) / fs
t_corr -= .082
fig, ax = plt.subplots()
for ch, l in zip(all_corr, ['ch 1: right', 'ch 2: left']):
	ax.plot(t_corr * 1e3, np.abs(ch), label=l)
#ax.plot(t_corr * 1e3, all_corr.T)
ax.set_xlim(0, 30)
ax.set_ylim(0, 350)
ax.set_xlabel('time, ms')
ax.set_ylabel('signal magnitude')
ax.grid()
ax.legend()

plt.show(block=False)
