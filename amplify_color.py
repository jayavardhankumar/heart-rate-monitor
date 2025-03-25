import cv2
import numpy as np

class AmplifyColor:
    """Applies Eulerian video magnification for subtle color changes in the ROI."""

    def __init__(self, alpha=50, levels=3):
        """Initializes color amplification parameters.
        
        Args:
            alpha (float): Amplification factor for color changes.
            levels (int): Number of pyramid levels for processing.
        """
        self.alpha = alpha
        self.levels = levels

    def amplify(self, frame):
        """Enhances subtle color changes in the input frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Color-amplified frame.
        """
        if frame is None or frame.size == 0:
            return frame

        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(frame_ycrcb)

        # Apply Gaussian pyramid downsampling
        for _ in range(self.levels):
            y = cv2.pyrDown(y)

        for _ in range(self.levels):
            y = cv2.pyrUp(y)

        # Amplify the color changes
        amplified_y = np.clip(y * self.alpha, 0, 255).astype(np.uint8)
        result = cv2.merge([amplified_y, cr, cb])

        return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)
