from math import pi
import numpy as np
import matplotlib.pyplot as plt
from note_xmitt import NarrowBand
from note_processing import NoteProcessing
from scipy.signal import kaiser, resample

from scipy.interpolate import interp1d

cmap = plt.cm.magma_r
cmap.set_under('w')

fc = 10000
bw = 500
f_bb = 750
fs = 96000

# virtual array
d = 0.5
c = 1500.

# virtual arrival
theta_inc = 5.  # incident angle, degrees

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

xmitt = kaiser(pulse.num_samples, 1.0 * np.pi)
xmitt *= np.sin(2 * pi * 1000 * pulse.time)
data_in[2 * pulse.num_samples: 3 * pulse.num_samples, 0] = xmitt

# interpolate to create second channel arrival delayed from first
tau = np.sin(np.radians(theta_inc)) * d / c
tau = np.round(tau * fs) / fs
taui = int(tau * fs)

data_in[2 * pulse.num_samples + taui: 3 * pulse.num_samples + taui, 1] = xmitt

# construct delay vector for beamforming
theta_axis = np.arange(num_theta) / num_theta * pi - pi / 2
tau_beam = np.zeros((num_theta, 2), dtype=np.float_)
tau_beam[:, 1] = np.sin(theta_axis) * d / c

sig_up = resample(xmitt, 10 * pulse.num_samples)
taxis_up = np.arange(pulse.num_samples * 10) / (fs * 10)

sig_ier = interp1d(taxis_up + (2 * pulse.num_samples + taui) / fs, sig_up,
                   kind='cubic', axis=0, fill_value=0, bounds_error=False)

beam_out = np.tile(sig_up, (tau_beam.shape[0], 1))
beam_out = beam_out.T

beam_times = data_taxis[:, None] - tau_beam[:, 1]
delayed_chans = sig_ier(beam_times)

beam_out = delayed_chans + (data_in[:, 0])[:, None]
beam_dB = np.sum(beam_out ** 2, axis=0)
beam_dB = 10 * np.log10(beam_dB)
beam_dB -= np.max(beam_dB)

fig, ax = plt.subplots()
#ax.plot(data_taxis, data_in[:, 0])
#ax.plot(data_taxis, delayed_chans[:, 0])
#ax.plot(data_taxis, delayed_chans[:, -1])
#cm = ax.pcolormesh(theta_axis, data_taxis, beam_out ** 2, vmin=1, vmax=4, cmap=cmap)
ax.plot(theta_axis, beam_dB)
#fig.colorbar(cm)
plt.show()
