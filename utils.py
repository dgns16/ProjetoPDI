import matplotlib.pyplot as plt
import cv2

def show_histogram(image):
    plt.figure()
    plt.title("Histograma")
    plt.xlabel("Intensidade")
    plt.ylabel("Frequência")
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.show()
