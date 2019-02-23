import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.signal import kaiser

fs = 96e3
fc = 40e3
period = 4e-3

num_tsig = np.ceil(period * fs)
t_sig = np.arange(num_tsig) / fs

xmitt = np.sin(2 * pi * fc * t_sig)
# window is unknown, assuming a pretty narrow mainlobe
window = kaiser(num_tsig, 1.0 * np.pi)
xmitt *= window


faxis = np.arange(num_tsig // 2 + 1) / num_tsig * fs
xmitt_FT = np.fft.rfft(xmitt)

xmitt_dB = 20 * np.log10(np.abs(xmitt_FT))
xmitt_dB -= np.max(xmitt_dB)

fig, ax = plt.subplots()
ax.plot(faxis, xmitt_dB, '.')
ax.set_ylim(-40, 0)

plt.show(block=False)
