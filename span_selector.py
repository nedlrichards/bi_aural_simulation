import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
from cycler import cycler

class MultiLine:
    """Plot multiple arrivals sequentially, with renormalization option"""
    def __init__(self, t_bounds = (45, 65)):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax_reset = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bnext = Button(ax_reset, 'Reset')
        self.bnext.on_clicked(self.reset)
        self.reading = None
        self.t_bounds = t_bounds
        self.span = SpanSelector(self.ax, self.onselect, 'horizontal',
                                 rectprops=dict(alpha=0.5, facecolor='red'))
        self.reset(None)

    def __call__(self, reading):
        """Plot completed samples on the current graph"""
        self.reading = reading
        self._replot()
        plt.show(block=False)

    @property
    def all_max(self):
        t_bounds = slice(self.tmin, self.tmax)
        all_max = np.max(np.abs(self.reading.sel(time=t_bounds)))
        return float(all_max)

    def onselect(self, tmin, tmax):
        self.tmin = tmin / 1e3
        self.tmax = tmax / 1e3
        self._replot()

    def reset(self, event):
        """put plot back to old t limits"""
        self.tmin = self.t_bounds[0] * 1e-3
        self.tmax= self.t_bounds[1] * 1e-3  # worked OK so far
        if event is not None:
            self._replot()


    def _replot(self):
        """Plot a normalized time series"""
        self.ax.clear()
        self.ax.set_prop_cycle(cycler('color', ['b', 'g']))
        for i, cs in self.reading.groupby('sample'):
            self.ax.plot(cs.time * 1e3, np.abs(cs.T) / self.all_max + 0.5 * i)

        self.ax.set_xlim(self.tmin * 1e3, self.tmax * 1e3)
        self.ax.set_ylim(0, 3.5)
        self.fig.canvas.draw()
