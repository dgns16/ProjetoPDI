import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_fourier_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.title("Espectro de Fourier")
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')
    plt.show()

def fourier_filter(image, mode='low'):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)

    r = 30
    if mode == 'low':
        mask[crow - r:crow + r, ccol - r:ccol + r] = 1
    elif mode == 'high':
        mask[:] = 1
        mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
