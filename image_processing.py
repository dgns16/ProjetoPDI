import cv2
import numpy as np

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return image.copy()  # Evita divisão por zero (imagem de tom único)

    stretched = (image - min_val) * (255.0 / (max_val - min_val))
    return stretched.astype(np.uint8)


def apply_spatial_filter(image, filter_type):
    if filter_type == 'mean':
        return cv2.blur(image, (3, 3))
    elif filter_type == 'median':
        return cv2.medianBlur(image, 3)
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (3, 3), 0)
    elif filter_type == 'max':
        return cv2.dilate(image, np.ones((3, 3), np.uint8))
    elif filter_type == 'min':
        return cv2.erode(image, np.ones((3, 3), np.uint8))
    elif filter_type == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
    elif filter_type == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        sobel = np.hypot(sobelx, sobely)
        return np.uint8(sobel)
    elif filter_type == 'prewitt':
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(image, -1, kernelx)
        prewitty = cv2.filter2D(image, -1, kernely)
        return np.maximum(prewittx, prewitty)
    elif filter_type == 'roberts':
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(image, -1, kernelx)
        robertsy = cv2.filter2D(image, -1, kernely)
        return np.maximum(robertsx, robertsy)
    else:
        return image
