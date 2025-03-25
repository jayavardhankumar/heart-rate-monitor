import cv2
import dlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceDetection:
    """Handles face detection using dlib's face detector."""

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

            # Extract **Forehead ROI**
            forehead_x1 = x + int(0.3 * w)
            forehead_x2 = x + int(0.7 * w)
            forehead_y1 = y + int(0.1 * h)
            forehead_y2 = y + int(0.25 * h)
            forehead_roi = frame[forehead_y1:forehead_y2, forehead_x1:forehead_x2]

            # Extract **Left Cheek ROI**
            left_cheek_x1 = x + int(0.05 * w)
            left_cheek_x2 = x + int(0.3 * w)
            left_cheek_y1 = y + int(0.5 * h)
            left_cheek_y2 = y + int(0.75 * h)
            left_cheek_roi = frame[left_cheek_y1:left_cheek_y2, left_cheek_x1:left_cheek_x2]

            # Extract **Right Cheek ROI**
            right_cheek_x1 = x + int(0.7 * w)
            right_cheek_x2 = x + int(0.95 * w)
            right_cheek_y1 = y + int(0.5 * h)
            right_cheek_y2 = y + int(0.75 * h)
            right_cheek_roi = frame[right_cheek_y1:right_cheek_y2, right_cheek_x1:right_cheek_x2]

            # Store in dictionary
            rois["forehead"] = forehead_roi
            rois["left_cheek"] = left_cheek_roi
            rois["right_cheek"] = right_cheek_roi

            return faces, rois  # ✅ Always return 2 values

        return [], rois  # ✅ Empty list for faces & empty ROIs if no detection
