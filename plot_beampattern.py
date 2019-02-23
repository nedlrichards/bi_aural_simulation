import numpy as np
import matplotlib.pyplot as plt
from note_xmitt import NarrowBand
from note_processing import NoteProcessing

cmap = plt.cm.magma_r
cmap.set_under('w')


fc = 25000
bw = 250
f_bb = 1500
fs = 96000

# virtual array
c = 1500.
#d = 1.5 * c / fc
d = 0.01875

# virtual arrival
theta_inc = 0  # incident angle, degrees
num_theta = 600

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
tau_in = np.degrees(np.arcsin(tau * c / d))

data_in[9 * pulse.num_samples + taui: 10 * pulse.num_samples + taui, 1] = pulse.signal

# construct delay vector for beamforming
sin_axis = np.arange(num_theta) / num_theta * 2 - 1
tau_beam = np.zeros((num_theta, 2), dtype=np.float_)
tau_beam[:, 1] = sin_axis * d / c

# baseband simulated data
data_bb = processor.baseband(data_in)
beam_out = processor.beamform(data_bb, tau_beam)

beam_dB = 10 * np.log10(beam_out)
beam_dB -= np.max(beam_dB)

# find optimal phase
compi = np.argmax(beam_dB)

fig, ax = plt.subplots()
ax.plot(np.degrees(np.arcsin(sin_axis)), beam_dB[compi // num_theta, :])
ax.plot([tau_in] * 2, [0] * 2, '.')
ax.set_title('Beamformer output at largest arrival time')
ax.set_xlabel(r'$sin(\theta)$')
ax.set_ylabel('dB re max')
ax.grid()
ax.set_ylim(-10, 1)
ax.set_yticks([-9, -6, -3, 0])

plt.show(block=False)
