import numpy as np
import pyaudio as pa
import probe_signal, pa_np_interface, loop_buffer, echo_processor
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from io import BytesIO
import queue
from cycler import cycler

# Create a probe signal
fc = 7000  # Hz
bw = 12000  # Hz
fs = int(44.1e3)  # Hz
duty_cycle = 0.5
T = duty_cycle + 0.3  # second
num_cycles = 1  # seconds
rough_time = num_cycles * T + 0.1 # Specify recording parameters
lfm = probe_signal.LFM(duty_cycle, fc, bw, T, fs)
# Matched filter specifications
replica = probe_signal.LFM(duty_cycle, fc, bw, duty_cycle, fs).signal
f_bounds = (fc - bw / 2, fc + bw / 2)
sp_er = echo_processor.Processor(replica, f_bounds, fs)
# Recording specifications
samples_per_chirp = int(T * fs)

# input data
read_samples = int(2 ** 10)
looper = loop_buffer.LoopBuffer(lfm.signal)
chunck_size = int(read_samples * looper.num_out_channels * looper.num_bytes)

# output method
dc = pa_np_interface.Dechunker()

# play and record callback

q = queue.Queue()

record = []

def buffer_callback(lfm_out, in_data, frame_count, time_info, status):
    q.put([in_data])
    out_data = lfm_out.read()
    return (out_data, pa.paContinue)

# setup usb device
def find_UA101():
    """Find device with UA-101 in string"""
    for i in range(p.get_device_count()):
        name = p.get_device_info_by_index(i)['name']
        if 'UA-101' in name:
            return i

def run_loop():
    """Put pyaudio stuff in a try block :)"""
    elapsed_time = 0
    index_start = 0
    current_cycle = 0
    completed_cycles = []
    samples = np.zeros((looper.num_out_channels, samples_per_chirp),
                       dtype=np.float32)
    # make samples a power of 2 for ease of filtering
    start_time = time.time()
    while time.time() - start_time < rough_time:
        # Rely on callback to break loop
        time.sleep(0.1)
        while not q.empty():
            if index_start + read_samples < samples_per_chirp:
                next_samples = slice(index_start, index_start + read_samples)
                current_buffer = dc.buf_to_np(b''.join(q.get())).T
                samples[:, next_samples] = current_buffer
                index_start += read_samples
            else:
                amount_extra = int(index_start + read_samples -
                                   samples_per_chirp)
                amount_space = int(read_samples - amount_extra)
                samples[:, index_start: ] = current_buffer[:, :amount_space]
                result = samples.copy()
                # The first chuck of data may happen before X-mission has begun
                #result = result[:, slice(read_samples, None)]
                result = sp_er(result)
                # Save the rest of the buffer to the new samples array
                samples[:, :amount_extra] = current_buffer[:, amount_space:]
                completed_cycles.append(result)
                index_start = amount_extra

    return completed_cycles

try:
    p = pa.PyAudio()
    # start up loop
    looper(num_cycles, chunck_size)
    callback = lambda *args: buffer_callback(looper, *args)
    stream = p.open(format=pa.paInt24,
                    channels=looper.num_out_channels,
                    rate=int(fs),
                    output=True,
                    input=True,
                    output_device_index=find_UA101(),
                    input_device_index=find_UA101(),
                    stream_callback=callback,
                    frames_per_buffer=read_samples)
    stream.start_stream()
    completed_samples = run_loop()
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

fig, ax = plt.subplots()
#fig, ax1 = plt.subplots()
ax.set_prop_cycle(cycler('color', ['b', 'g']))
#ax1.set_prop_cycle(cycler('color', ['b', 'g']))
all_max = np.max(np.array([np.max(np.abs(cs)) for cs in completed_samples]))
for i, cs in enumerate(completed_samples):
    # Don't know why we are adding a buffer each time
    t_range = np.arange(cs.shape[1]) - read_samples * i
    dB = 20 * np.log10(np.abs(cs.T))
    dB -= 20 * np.log10(np.abs(all_max))
    ax.plot(t_range / fs * 1e3, np.abs(cs.T) / all_max + 0.5 * i)
ax.set_xlim(45, 65)
ax.grid()
ax.set_xlabel('time, ms')
ax.set_ylabel('amplitude')
ax.set_title('single echo measurement')

plt.show(block=False)
