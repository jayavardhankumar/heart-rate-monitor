import cv2
import dlib
import numpy as np
from imutils import face_utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceUtilities:
    def __init__(self):
        """Initialize face utilities, including Dlib's face detector and landmark predictor."""
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            logging.info("Face utilities initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing face utilities: {e}")
            raise

    def align_face(self, frame, landmarks):
        """Align the face using the position of the eyes."""
        try:
            left_eye_pts = landmarks[36:42]
            right_eye_pts = landmarks[42:48]
            left_eye_center = np.mean(left_eye_pts, axis=0).astype("int")
            right_eye_center = np.mean(right_eye_pts, axis=0).astype("int")
            
            dx, dy = right_eye_center - left_eye_center
            angle = np.degrees(np.arctan2(dy, dx))
            
            eyes_center = tuple(((left_eye_center + right_eye_center) // 2).astype(int))
            
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
            aligned_face = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            return aligned_face
        except Exception as e:
            logging.error(f"Error aligning face: {e}")
            return frame

    def extract_ROI(self, frame, landmarks):
        """Extract regions of interest (ROIs) from the forehead and cheeks."""
        try:
            forehead = frame[landmarks[19][1] - 30:landmarks[24][1] - 10, landmarks[19][0]:landmarks[24][0]]
            right_cheek = frame[landmarks[29][1]:landmarks[33][1], landmarks[54][0]:landmarks[12][0]]
            left_cheek = frame[landmarks[29][1]:landmarks[33][1], landmarks[4][0]:landmarks[48][0]]
            return forehead, left_cheek, right_cheek
        except Exception as e:
            logging.error(f"Error extracting ROIs: {e}")
            return None, None, None

    def process_face(self, frame):
        """Detect and process face alignment and ROI extraction."""
        faces = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if len(faces) == 0:
            return None, None, None, None
        
        for face in faces:
            landmarks = self.predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face)
            landmarks = face_utils.shape_to_np(landmarks)
            aligned_face = self.align_face(frame, landmarks)
            forehead, left_cheek, right_cheek = self.extract_ROI(aligned_face, landmarks)
            return aligned_face, forehead, left_cheek, right_cheek
        
        return None, None, None, None
