import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Webcam:
    """Handles webcam video capture for real-time heart rate monitoring."""

    def __init__(self, source=0):
        """Initializes webcam capture.

        Args:
            source (int): Index of the webcam (default: 0).
        """
        self.source = source
        self.capture = None
        self.is_running = False

    def start(self):
        """Starts the webcam capture."""
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            logging.error("[Webcam] Failed to open webcam.")
            return False
        
        self.is_running = True
        logging.info("[Webcam] Webcam started successfully at index %d", self.source)
        return True

    def read_frame(self):
        """Reads a frame from the webcam.

        Returns:
            np.ndarray or None: The captured frame or None if reading failed.
        """
        if not self.is_running:
            logging.warning("[Webcam] Webcam is not running.")
            return None

        ret, frame = self.capture.read()
        if not ret:
            logging.warning("[Webcam] Failed to read frame.")
            return None
        
        return frame

    def stop(self):
        """Stops the webcam capture."""
        if self.capture:
            self.capture.release()
        self.is_running = False
        logging.info("[Webcam] Webcam stopped.")
