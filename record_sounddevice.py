import sounddevice as sd
import numpy as np
from math import pi
from queue import Queue
import matplotlib.pyplot as plt
from ua101_node import UA101

fc = 5000  # Hz
record_threshold = 0.5  # begin recording when this value is exceeded
ua = UA101(fc, record_threshold)

fig, ax = plt.subplots()
plt.plot(ua.taxis, np.abs(ua.recorded_data[:, 0]))
plt.plot(ua.taxis, np.abs(ua.recorded_data[:, 1]))

plt.show(block=False)
