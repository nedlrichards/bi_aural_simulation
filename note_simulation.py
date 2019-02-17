from math import pi
import numpy as np
import matplotlib.pyplot as plt
from note_xmitt import NarrowBand
from note_processing import NoteProcessing

cmap = plt.cm.magma_r
cmap.set_under('w')

fc = 5000
bw = 500
f_bb = 750
fs = 96000

# virtual array
d = 0.5
c = 1500.

# virtual arrival
theta_inc = 35.  # incident angle, degrees

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
data_in[2 * pulse.num_samples: 3 * pulse.num_samples, 0] = pulse.signal

# interpolate to create second channel arrival delayed from first
tau = np.sin(np.degrees(theta_inc)) * d / c
data_in[:, 1] = np.interp(data_taxis - tau, data_taxis, data_in[:, 0],
                          left=0, right=0)

# construct delay vector for beamforming
theta_axis = np.arange(num_theta) / num_theta * np.pi - np.pi / 2
tau_beam = np.zeros((num_theta, 2), dtype=np.float_)
tau_beam[:, 1] = np.sin(theta_axis) * d / c

# baseband simulated data
data_bb = processor.baseband(data_in)
beam_out = processor.beamform(data_bb, tau_beam)

beam_dB = 20 * np.log10(np.abs(beam_out))
beam_dB -= np.max(beam_dB)

fig, ax = plt.subplots()
cm = ax.pcolormesh(processor.taxis * 1e3,
                   np.degrees(theta_axis), beam_dB.T,
                   vmin=-10, vmax=0, cmap=cmap)
ax.set_ylabel('theta, $^o$')
ax.set_xlabel('arrival time, ms')
ax.grid()
fig.colorbar(cm)
ax.set_xlim(5, 20)

plt.show(block=False)
1/0


in_data_FT = np.fft.fft(data_in, axis=0)
in_data_FT *= processor.bp_FT[:, None]
recorded_data_bb = np.fft.ifft(in_data_FT, axis=0)

f_shift = fc - f_bb
phase_bb = np.exp(-2j * pi * f_shift * data_taxis)[:, None]
recorded_data_bb *= phase_bb

in_data_dB = 20 * np.log10(np.abs(in_data_FT))
in_data_dB -= np.max(in_data_dB)

fig, ax = plt.subplots()
#ax.plot(data_taxis * 1e3, np.real(recorded_data_bb))
ax.plot(processor.taxis * 1e3, np.abs(data_bb))
ax.set_xlabel('time, ms')
ax.set_ylabel('amplitude')
ax.set_title('Transmitted pulse')
ax.grid()
plt.show(block=False)
1/0



fig, ax = plt.subplots()
ax.plot(data_faxis / 1e3, in_data_dB)
ax.set_xlim(0, 10)
ax.set_ylim(-60, 3)
ax.set_xlabel('Frequency, kHz')
ax.set_ylabel('magnitude, dB re max')
ax.set_title('Transmitted pulse')
ax.grid()

fig, ax = plt.subplots()
ax.plot(pulse.time * 1e3, pulse.signal)
ax.set_xlabel('time, ms')
ax.set_ylabel('amplitude')
ax.set_title('Transmitted pulse')
ax.grid()

fig, ax = plt.subplots()
ax.plot(pulse_faxis / 1e3, pulse_dB)
ax.set_xlim(0, 10)
ax.set_ylim(-60, 3)
ax.set_xlabel('Frequency, kHz')
ax.set_ylabel('magnitude, dB re max')
ax.set_title('Transmitted pulse')
ax.grid()

plt.show(block=False)
