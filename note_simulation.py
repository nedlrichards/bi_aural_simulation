from math import pi
import numpy as np
import matplotlib.pyplot as plt
from note_xmitt import NarrowBand
from note_processing import NoteProcessing
from scipy.signal import kaiser, resample

from scipy.interpolate import interp1d

cmap = plt.cm.magma_r
cmap.set_under('w')

fc = 40000
bw = 500
f_bb = 1500
fs = 96000

# virtual array
d = 0.5
c = 1500.

# virtual arrival
theta_inc = 25  # incident angle, degrees

# beamformer specification
num_theta = 300

pulse = NarrowBand(fc, bw, fs)

processor = NoteProcessing(fc, f_bb, 0.5, fs)

# compute FFT of signal
NFFT = int(2 ** np.ceil(np.log2(pulse.signal.size) + 4))
pulse_FT = np.fft.rfft(pulse.signal, NFFT)
pulse_faxis = np.arange(NFFT // 2 + 1) / NFFT * fs

pulse_dB = 20 * np.log10(np.abs(pulse_FT))
pulse_dB -= np.max(pulse_dB)

# create a simulated time series for baseband processing
data_taxis = processor._taxis
data_faxis = np.arange(data_taxis.size) / data_taxis.size * fs
data_in = np.zeros((data_taxis.size, 2), dtype=np.float_)
data_in[9 * pulse.num_samples: 10 * pulse.num_samples, 0] = pulse.signal

# interpolate to create second channel arrival delayed from first
tau = np.sin(np.radians(theta_inc)) * d / c
tau = np.round(tau * fs) / fs
taui = int(tau * fs)

data_in[9 * pulse.num_samples + taui: 10 * pulse.num_samples + taui, 1] = pulse.signal

# construct delay vector for beamforming
theta_axis = np.arange(num_theta) / num_theta * pi - pi / 2
tau_beam = np.zeros((num_theta, 2), dtype=np.float_)
tau_beam[:, 1] = np.sin(theta_axis) * d / c

# baseband simulated data
data_bb = processor.baseband(data_in)
beam_out = processor.beamform(data_bb, tau_beam)

beam_dB = 10 * np.log10(beam_out)
beam_dB -= np.max(beam_dB)

fig, ax = plt.subplots()
cm = ax.pcolormesh(processor.taxis[-beam_dB.shape[0]:] * 1e3,
                   np.degrees(theta_axis),
                   beam_dB.T, vmin=-3, vmax=0, cmap=cmap)

ax.set_xlabel('time, ms')
ax.set_ylabel('theta, $^o$')
ax.set_xlim(0, 50)
ax.grid()
fig.colorbar(cm)

# find the largest value
mi = np.argmax(beam_dB)
theta_max = np.degrees(theta_axis[mi % 300])

plt.show(block=False)
