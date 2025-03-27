import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

class GraphPlot:
    """Handles real-time plotting of signal and FFT data using PyQtGraph, with ROI preview support."""

    def __init__(self, signal_plot, fft_plot, roi_label: QLabel):
        """Initializes graph plotting elements.

        Args:
            signal_plot (pg.PlotWidget): Widget for displaying the raw signal.
            fft_plot (pg.PlotWidget): Widget for displaying the FFT spectrum.
            roi_label (QLabel): QLabel widget for displaying the ROI preview.
        """
        self.signal_plot = signal_plot
        self.fft_plot = fft_plot
        self.roi_label = roi_label  # QLabel to display ROI preview

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

    def update_roi_preview(self, roi):
        """Updates the ROI preview above the signal graph.

        Args:
            roi (np.ndarray): Extracted ROI image.
        """
        if roi is not None:
            roi_resized = cv2.resize(roi, (100, 50))  # Resize for better visibility
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB format

            # Convert OpenCV image to Qt image
            height, width, channel = roi_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(roi_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Update QLabel with new ROI preview
            self.roi_label.setPixmap(QPixmap.fromImage(q_image))
