import cv2
import numpy as np

def apply_erosion(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    return eroded

def apply_dilation(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated
