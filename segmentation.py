import cv2

def apply_otsu_threshold(image):
    # Aplica limiarização usando o método de Otsu
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
