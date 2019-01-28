import sounddevice as sd
import numpy as np
from math import pi
from queue import Queue
import matplotlib.pyplot as plt
from ua101_node import UA101

fc = 5000  # Hz
record_threshold = 0.5  # begin recording when this value is exceeded
ua = UA101(fc, record_threshold)
1/0

# setup xaxis



# Wait untill record



bbt, bbout = bber(recorded_data)

fig, ax = plt.subplots()
#plt.plot(bbt, data_bp[:, 0])
plt.plot(np.arange(recorded_data.shape[0]) / fs, recorded_data[:, 0])
plt.plot(bbt, np.abs(bbout[:, 0]))
plt.plot(bbt, np.abs(bbout[:, 1]))
#plt.plot(bbt, np.abs(bbout[:, 1]), color='C2')
#plt.plot(bbt, np.real(bb_td[:, 0]))
#plt.plot(bbt, np.imag(bb_td[:, 0]))

plt.show(block=False)
