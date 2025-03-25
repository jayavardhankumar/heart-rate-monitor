import cv2
import numpy as np

def resize(*args, **kwargs):
    """Resize an image."""
    return cv2.resize(*args, **kwargs)

def moveWindow(*args, **kwargs):
    """Move a window (Placeholder function)."""
    return

def imshow(*args, **kwargs):
    """Show an image using OpenCV."""
    return cv2.imshow(*args, **kwargs)

def destroyWindow(*args, **kwargs):
    """Destroy a specific OpenCV window."""
    return cv2.destroyWindow(*args, **kwargs)

def waitKey(*args, **kwargs):
    """Wait for a key press in OpenCV."""
    return cv2.waitKey(*args, **kwargs)

def combine(left, right):
    """Stack images horizontally."""
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    
    comb = np.zeros((h, w, 3), left.dtype)
    comb[:left.shape[0], :left.shape[1]] = left
    comb[:right.shape[0], left.shape[1]:] = right
    
    return comb
