import numpy as np
import scipy.signal as signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SignalProcessing:
    """Handles heart rate estimation using signal processing techniques."""

    def __init__(self, buffer_size=250, sampling_rate=30):
        """Initializes signal processing for heart rate estimation.

        Args:
            buffer_size (int): Number of frames stored for processing.
            sampling_rate (int): Frame rate of the video (default 30 FPS).
        """
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.samples = []
        self.freqs = []
        self.fft_values = []
        self.bpms = []
        logging.info("[SignalProcessing] Initialized with buffer size: %d, sampling rate: %d", buffer_size, sampling_rate)

    def add_sample(self, value):
        """Adds a new sample to the signal buffer.

        Args:
            value (float): Extracted intensity value from ROI.
        """
        self.samples.append(value)
        if len(self.samples) > self.buffer_size:
            self.samples.pop(0)

    def compute_fft(self):
        """Computes the FFT of the current signal buffer."""
        if len(self.samples) < self.buffer_size:
            logging.warning("[SignalProcessing] Not enough samples for FFT computation.")
            return None

        signal_array = np.array(self.samples, dtype=np.float32)
        detrended_signal = signal.detrend(signal_array)  # Remove trends to get better frequency response
        fft_result = np.abs(np.fft.rfft(detrended_signal))
        freqs = np.fft.rfftfreq(len(detrended_signal), d=1.0 / self.sampling_rate)

        self.freqs = freqs
        self.fft_values = fft_result

        return freqs, fft_result

    def estimate_heart_rate(self):
        """Estimates the heart rate from the frequency spectrum.

        Returns:
            bpm (float): Estimated heart rate in beats per minute.
        """
        if len(self.fft_values) == 0 or len(self.freqs) == 0:
            logging.warning("[SignalProcessing] FFT values are empty, cannot estimate heart rate.")
            return None

        min_bpm = 60
        max_bpm = 110
        min_freq = min_bpm / 60.0
        max_freq = max_bpm / 60.0

        valid_indices = np.where((self.freqs >= min_freq) & (self.freqs <= max_freq))
        valid_freqs = self.freqs[valid_indices]
        valid_fft_values = self.fft_values[valid_indices]

        if len(valid_fft_values) == 0:
            logging.warning("[SignalProcessing] No valid frequency detected within heart rate range.")
            return None

        peak_index = np.argmax(valid_fft_values)
        peak_freq = valid_freqs[peak_index]
        estimated_bpm = peak_freq * 60.0

        self.bpms.append(estimated_bpm)
        if len(self.bpms) > self.buffer_size:
            self.bpms.pop(0)

        logging.info("[SignalProcessing] Estimated Heart Rate: %.2f BPM", estimated_bpm)
        return estimated_bpm

    def reset(self):
        """Resets the signal processing buffers."""
        self.samples.clear()
        self.bpms.clear()
        self.freqs.clear()
        self.fft_values.clear()
        logging.info("[SignalProcessing] Reset buffers.")
