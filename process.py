import cv2
import numpy as np
import logging
from face_detection import FaceDetection
from signal_processing import SignalProcessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Process:
    """Handles the real-time processing pipeline."""

    def __init__(self):
        """Initializes face detection and signal processing."""
        self.face_detector = FaceDetection()
        self.signal_processor = SignalProcessing(buffer_size=250, sampling_rate=30)
        self.video_capture = None
        self.is_running = False
        logging.info("[Process] Initialized.")

    def start_webcam(self):
        """Starts the webcam stream."""
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            logging.error("[Process] Webcam failed to start.")
            return False
        self.is_running = True
        logging.info("[Process] Webcam started successfully.")
        return True

    def start_video(self, video_path):
        """Starts video file processing."""
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            logging.error("[Process] Failed to open video file.")
            return False
        self.is_running = True
        logging.info(f"[Process] Video {video_path} loaded successfully.")
        return True

    def stop(self):
        """Stops the webcam or video stream."""
        if self.video_capture:
            self.video_capture.release()
        self.is_running = False
        logging.info("[Process] Video/Webcam stopped.")

    def get_frame(self):
        """Captures a frame from the webcam or video."""
        if not self.is_running:
            return None

        ret, frame = self.video_capture.read()
        if not ret:
            logging.error("[Process] Failed to capture frame.")
            return None
        return frame

    def run(self, frame):
        """Processes the frame for face detection and heart rate estimation."""
        if frame is None:
            logging.error("[Process] No frame received!")
            return frame, None, 0, [], [], []

        # Detect faces and extract ROIs
        faces, rois = self.face_detector.detect_face(frame)
        if len(faces) == 0:
            logging.warning("[Process] No face detected.")
            return frame, None, 0, [], [], []

        # Select the largest face for stability
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # **Extract Forehead and Cheeks as ROI**
        forehead_roi = frame[y:y + int(h * 0.2), x + int(w * 0.3):x + int(w * 0.7)]
        left_cheek_roi = frame[y + int(h * 0.5):y + int(h * 0.7), x:x + int(w * 0.2)]
        right_cheek_roi = frame[y + int(h * 0.5):y + int(h * 0.7), x + int(w * 0.8):x + w]

        # Ensure ROIs are valid before processing
        if forehead_roi.size == 0 or left_cheek_roi.size == 0 or right_cheek_roi.size == 0:
            logging.warning("[Process] Invalid ROI detected.")
            return frame, None, 0, [], [], []

        # Convert ROI to grayscale
        roi_gray = cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(roi_gray)

        # Process signal
        self.signal_processor.add_sample(mean_intensity)
        signal_data = self.signal_processor.samples
        fft_result = self.signal_processor.compute_fft()

        if fft_result is None:
            freqs, fft_values = [], []
        else:
            freqs, fft_values = fft_result

        bpm = self.signal_processor.estimate_heart_rate() or 0

        # **Draw Face & ROI rectangles**
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Face
        cv2.rectangle(frame, (x + int(w * 0.3), y), (x + int(w * 0.7), y + int(h * 0.2)), (255, 0, 0), 2)  # Forehead
        cv2.rectangle(frame, (x, y + int(h * 0.5)), (x + int(w * 0.2), y + int(h * 0.7)), (0, 0, 255), 2)  # Left cheek
        cv2.rectangle(frame, (x + int(w * 0.8), y + int(h * 0.5)), (x + w, y + int(h * 0.7)), (0, 0, 255), 2)  # Right cheek

        logging.info("[Process] Estimated Heart Rate: %.2f BPM", bpm)
        return frame, forehead_roi, bpm, signal_data, freqs, fft_values
