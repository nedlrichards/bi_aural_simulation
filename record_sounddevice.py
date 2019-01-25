import sounddevice as sd
import numpy as np
from math import pi
from queue import Queue
import matplotlib.pyplot as plt
from scipy.signal import firwin

fc = 5000  # Hz
bw = 500   # Hz

NFFT = 8192  # a power of 2 is nice
blocksize = NFFT // 2  # This line of code should not change!
numblocks = 16

record_threshold = 0.5  # begin recording when this value is exceeded
is_exceed = False  # flag that signals exceedence

# find UA-101
all_devs = sd.query_devices()
ua_101_dev = [d for d in all_devs if 'UA-101' in d['name']]
if len(ua_101_dev) == 1:
    sd.default.device = ua_101_dev[0]['name']
else:
    raise(ValueError('UA-101 device not properly identified'))

# set remaining default information
sd.default.channels = 2
sd.default.samplerate = int(ua_101_dev[0]['default_samplerate'])
fs = sd.default.samplerate
sd.default.blocksize = NFFT // 2

# design baseband filter
f_bb = 3 * bw
bp_filt = firwin(blocksize // 2 + 1, [f_bb - 2 * bw, f_bb + 2 * bw],
                 window='blackmanharris', pass_zero=False, fs=fs)

# setup xaxis
faxis = np.arange(NFFT // 2 + 1) / NFFT * fs
fci = np.bitwise_and(faxis > fc - bw, faxis < fc + bw)

start_time = None
elapsed_time = None
lastdata = Queue()
all_data = Queue()

def callback(indata, outdata, frames, time, status):
    global is_exceed

    if status:
        print(status)


    outdata.fill(0)
    curr_data = indata.copy()
    curr_data = curr_data.astype(np.float64)

    if is_exceed:
        all_data.put(curr_data)
        if all_data.qsize() >= numblocks - 1:
            raise(sd.CallbackStop)
        return

    # special case for first sample
    if lastdata.empty():
        lastdata.put(curr_data)
        return

    # construct array of this sample and last
    full_data = np.concatenate([lastdata.get(), curr_data])
    lastdata.put(curr_data)

    # process last 2 samples
    data_FT = np.fft.rfft(full_data[:, 0])
    is_exceed = np.any(np.abs(data_FT[fci]) > record_threshold)

    if is_exceed:
        all_data.put(full_data)

with sd.Stream(dtype='float32', callback=callback) as s:
    while s.active:
        sd.sleep(int(100))

recorded_data = []
while not all_data.empty():
    recorded_data.append(all_data.get())

recorded_data = np.concatenate(recorded_data)

# plot bandpass filter and data
long_NFFT = recorded_data.shape[0]
faxis = np.arange(long_NFFT) / long_NFFT * fs
bp_FT = np.fft.fft(bp_filt, n=long_NFFT)
taxis = np.arange(recorded_data.shape[0]) / fs
complex_baseband = recorded_data \
                 * np.exp(-1j * 2 * pi * (fc - f_bb) * taxis[:, None])
#data_FT = np.fft.rfft(recorded_data, n=long_NFFT, axis=0)
data_FT = np.fft.fft(complex_baseband, n=long_NFFT, axis=0)
data_BB = bp_FT[:, None] * data_FT
data_BB = np.fft.ifft(data_BB, axis=0)

# downsample rate
f_max = f_bb + 3 * bw  # conservative estimate
dec_down = int(np.ceil(fs / f_max))
data_final = data_BB[::dec_down, :]
taxis_final = taxis[:: dec_down]
dt_final = (taxis_final[-1] - taxis_final[0]) / (taxis_final.size - 1)
faxis_final = np.arange(data_final.shape[0]) / data_final.shape[0] / dt_final

fig, ax = plt.subplots()
ax.plot(faxis, np.abs(np.fft.fft(data_BB[:, 0])))
ax.plot(faxis_final, np.abs(np.fft.fft(data_final[:, 0])))
#ax.plot(taxis, data_BB[:, 1], 'C2')

plt.show(block=False)
