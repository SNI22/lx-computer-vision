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
    steer_matrix_left = np.zeros(shape, dtype=np.float32)
    
    # Apply a gradient to emphasize the left side more strongly
    for i in range(shape[0]):
        for j in range(shape[1] // 2):
            # Higher weights on the far left, linearly decreasing towards the center
            steer_matrix_left[i, j] = 1 - (j / (shape[1] // 2))
    

    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    steer_matrix_right = np.zeros(shape, dtype=np.float32)
    
    # Apply a gradient to emphasize the right side more strongly
    for i in range(shape[0]):
        for j in range(shape[1] // 2, shape[1]):
            # Higher weights on the far right, linearly decreasing towards the center
            steer_matrix_right[i, j] = (j - shape[1] // 2) / (shape[1] // 2)
    return steer_matrix_right


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
    horizon_row = 175
    mask_ground[:horizon_row, :] = 0 
    # Apply color transformation matrix
    M = np.array([
        [0.5, 0, 0],
        [0, 2.55, 0],
        [0, 0, 2.55]
    ])
    white_lower_hsv = np.dot(M, np.array([0, 0, 49]))
    white_upper_hsv = np.dot(M, np.array([250, 15, 100]))
    yellow_lower_hsv = np.dot(M, np.array([34, 38, 66]))
    yellow_upper_hsv = np.dot(M, np.array([57, 105, 85]))

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

    # Edge direction and magnitude thresholds
    threshold = 70
    mask_mag = (Gmag > threshold).astype(np.uint8)
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_neg = (sobely < 0)
    
    # Combine masks to isolate left and right lane markings
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
