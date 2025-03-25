import pyqtgraph as pg
import numpy as np

class GraphPlot:
    """Handles real-time plotting of signal and FFT data using PyQtGraph."""

    def __init__(self, signal_plot, fft_plot):
        """Initializes graph plotting elements.

        Args:
            signal_plot (pg.PlotWidget): Widget for displaying the raw signal.
            fft_plot (pg.PlotWidget): Widget for displaying the FFT spectrum.
        """
        self.signal_plot = signal_plot
        self.fft_plot = fft_plot

        # Configure signal plot
        self.signal_plot.setBackground("black")
        self.signal_plot.setTitle("Raw Signal", color="w", size="12pt")
        self.signal_curve = self.signal_plot.plot(pen=pg.mkPen("g", width=2))

        # Configure FFT plot
        self.fft_plot.setBackground("black")
        self.fft_plot.setTitle("FFT Spectrum", color="w", size="12pt")
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen("r", width=2))

    def update_signal(self, signal_data):
        """Updates the signal graph with new data.

        Args:
            signal_data (list or np.ndarray): Signal intensity values.
        """
        if len(signal_data) > 0:
            self.signal_curve.setData(signal_data)

    def update_fft(self, freqs, fft_values):
        """Updates the FFT graph with new data.

        Args:
            freqs (list or np.ndarray): Frequency values.
            fft_values (list or np.ndarray): FFT magnitudes.
        """
        if len(freqs) > 0 and len(fft_values) > 0:
            self.fft_curve.setData(freqs, fft_values)
