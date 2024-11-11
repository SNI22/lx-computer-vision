from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """
    steer_left = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Quadratic increase for stronger left influence
            steer_left[i, j] = (j / (shape[1] - 1)) # Non-linear left influence
    return -0.7 * steer_left  # Amplify left influence slightly with -1.2 factor

def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """
    # steer_matrix_right = np.zeros(shape, dtype=np.float32)
    
    # Apply a gradient to emphasize the right side with symmetric weighting
    steer_right = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            steer_right[i, j] = (1 - (j / (shape[1] - 1)))  # Decreases from left to right (high values on the left)
    
    return 0.7*steer_right



def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape
    sigma = 1

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Mask out the ground up to the horizon line
    mask_ground = np.ones(image.shape[:2], dtype=np.uint8)
    horizon_row = 260
    mask_ground[:horizon_row, :] = 0 
    # Apply color transformation matrix
    M = np.array([
        [0.5, 0, 0],
        [0, 2.55, 0],
        [0, 0, 2.55]
    ])
    white_lower_hsv = np.dot(M, np.array([0, 0, 49]))
    white_upper_hsv = np.dot(M, np.array([250, 15, 100]))
    yellow_lower_hsv = np.dot(M, np.array([34, 22, 58]))
    yellow_upper_hsv = np.dot(M, np.array([65, 105, 105]))
    # white_lower_hsv = np.array([0, 0, 100])         # CHANGE ME
    # white_upper_hsv = np.array([230, 100, 255])   # CHANGE ME
    # yellow_lower_hsv = np.array([15, 50, 50])        # CHANGE ME
    # yellow_upper_hsv = np.array([50, 255, 255])  # CHANGE ME

    # Apply color masks
    mask_white = cv2.inRange(hsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(hsv, yellow_lower_hsv, yellow_upper_hsv)

    # Gaussian blur to reduce noise before edge detection
    img_gaussian_filter = cv2.GaussianBlur(gray_img, (0, 0), sigma)

    # Sobel gradients for edge detection
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    width = w

    # Define left and right region masks
    mask_left = np.ones(sobelx.shape[:2])
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape[:2])   
    mask_right[:,0:int(np.floor(width/2))] = 0

    mask_left = np.ones((h, w), dtype=np.uint8)
    mask_left[:, int(w * 0.8):] = 0  # Mask out the right 20%

    mask_right = np.ones((h, w), dtype=np.uint8)
    mask_right[:, :int(w * 0.3)] = 0  # Mask out the left 30%
    # Edge direction and magnitude thresholds
    threshold = 70
    mask_mag = (Gmag > threshold).astype(np.uint8)
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    
    # Combine masks to isolate left and right lane markings
    mask_left_edge = mask_ground  * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
