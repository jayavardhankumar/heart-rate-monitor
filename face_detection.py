import cv2
import dlib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceDetection:
    """Handles face detection using dlib's face detector and extracts ROIs for forehead and cheeks."""

    def __init__(self):
        """Initialize the face detector."""
        self.detector = dlib.get_frontal_face_detector()
        logging.info("[FaceDetection] Face detection model loaded successfully.")

    def detect_face(self, frame):
        """Detects faces and extracts ROIs for forehead and cheeks.

        Args:
            frame (numpy.ndarray): The input image frame.

        Returns:
            tuple: (faces, rois) where:
                - faces: List of detected face bounding boxes [(x, y, w, h), ...]
                - rois: Dictionary with extracted forehead and cheek regions
                    {"forehead": forehead_roi, "left_cheek": left_cheek_roi, "right_cheek": right_cheek_roi}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector(gray)

        faces = []
        rois = {"forehead": None, "left_cheek": None, "right_cheek": None}

        if len(detections) > 0:
            for d in detections:
                x, y, w, h = d.left(), d.top(), d.width(), d.height()
                faces.append((x, y, w, h))

            # Select the largest detected face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            # Define ROI coordinates safely
            def get_roi(x1, x2, y1, y2):
                """Returns a valid ROI ensuring it does not go out of frame bounds."""
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
                return roi if roi.size > 0 else None

            # Extract **Forehead ROI**
            forehead_roi = get_roi(
                x + int(0.3 * w), x + int(0.7 * w),
                y + int(0.1 * h), y + int(0.25 * h)
            )

            # Extract **Left Cheek ROI**
            left_cheek_roi = get_roi(
                x + int(0.05 * w), x + int(0.3 * w),
                y + int(0.5 * h), y + int(0.75 * h)
            )

            # Extract **Right Cheek ROI**
            right_cheek_roi = get_roi(
                x + int(0.7 * w), x + int(0.95 * w),
                y + int(0.5 * h), y + int(0.75 * h)
            )

            # Store in dictionary
            rois["forehead"] = forehead_roi
            rois["left_cheek"] = left_cheek_roi
            rois["right_cheek"] = right_cheek_roi

            return faces, rois  # ✅ Always return 2 values

        return [], rois  # ✅ Empty list for faces & empty ROIs if no detection
