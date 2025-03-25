import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Video:
    """Handles video file playback for heart rate analysis."""

    def __init__(self):
        """Initializes video playback settings."""
        self.video_path = None
        self.capture = None
        self.is_running = False

    def start(self, video_path):
        """Starts video playback.

        Args:
            video_path (str): Path to the video file.

        Returns:
            bool: True if the video started successfully, False otherwise.
        """
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            logging.error("[Video] Failed to open video file: %s", video_path)
            return False
        
        self.is_running = True
        logging.info("[Video] Video file loaded successfully: %s", video_path)
        return True

    def read_frame(self):
        """Reads a frame from the video file.

        Returns:
            np.ndarray or None: The captured frame or None if the video ends.
        """
        if not self.is_running:
            logging.warning("[Video] Video playback is not active.")
            return None

        ret, frame = self.capture.read()
        if not ret:
            logging.info("[Video] Video playback completed.")
            self.stop()
            return None
        
        return frame

    def stop(self):
        """Stops video playback."""
        if self.capture:
            self.capture.release()
        self.is_running = False
        logging.info("[Video] Video playback stopped.")
