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
num_cycles = 3  # seconds
rough_time = num_cycles * T + 0.1
# Specify recording parameters
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
ua = ua101_interface.UA101()

# play and record callback

q = queue.Queue()
record = []

def buffer_callback(lfm_out, in_data, frame_count, time_info, status):
    q.put([in_data])
    out_data = lfm_out.read()
    return (out_data, pa.paContinue)

def run_loop():
    """Put pyaudio stuff in a try block :)"""
    elapsed_time = 0
    index_start = 0
    current_cycle = 0
    completed_cycles = []
    samples = []
    # make samples a power of 2 for ease of filtering
    start_time = time.time()
    while time.time() - start_time < rough_time:
        # Rely on callback to break loop
        time.sleep(0.1)
        while not q.empty() and len(completed_cycles) < num_cycles:
            current_buffer = dc.buf_to_np(b''.join(q.get()))
            samples.append(current_buffer)
    return samples

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
                    output_device_index=ua.find_UA101(),
                    input_device_index=ua.find_UA101(),
                    stream_callback=callback,
                    frames_per_buffer=read_samples)
    stream.start_stream()
    completed_samples = run_loop()
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', ['b', 'g']))

def plot_stacked(sample_list):
    """a waterfall plot"""
    all_max = np.max(np.array([np.max(np.abs(cs)) for cs in sample_list]))
    # make 0 time max arrival
    max_time = np.argmax(sample_list[0][0, :]) / fs * 1e3
    # inter-line spaceing
    ils = 0.3
    for i, cs in enumerate(sample_list):
        # Don't know why we are adding a buffer each time
        t_range = (np.arange(cs.shape[1]) - read_samples * i) / fs * 1e3 - max_time
        ax.plot(t_range, np.abs(cs.T) / all_max + ils * i)
    ax.set_xlim(0, 0 + 20)
    ax.set_ylim(0, len(sample_list) * ils + ils / 3)
    ax.set_xlabel('time, ms')
    ax.set_ylabel('linear pressure scale')
    ax.grid()

def plot_db(one_sample):
    """A db plot, suited for a simple sample"""
    all_max = np.max(np.abs(one_sample))
    # make 0 time max arrival
    max_time = np.argmax(one_sample[0, :]) / fs * 1e3
    t_range = (np.arange(one_sample.shape[1])) / fs * 1e3 - max_time
    sample_db = 20 * np.log10(np.abs(one_sample.T))
    ax.plot(t_range, sample_db)
    ax.set_xlim(0, 0 + 20)
    ax.set_ylim(-20, 30)
    ax.set_xlabel('time, ms')
    ax.set_ylabel('log pressure scale')
    ax.grid()

if num_cycles == 1: plot_db(completed_samples[0])
else: plot_stacked(completed_samples)
plt.show()
