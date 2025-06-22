import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from image_processing import *
from fourier_utils import *
from morphology import *
from segmentation import *
from utils import *

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Imagens")
        self.root.state('zoomed')  # Janela em tela cheia (Windows)

        self.image = None
        self.gray_image = None
        self.undo_stack = []

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.menu = tk.Menu(root)
        root.config(menu=self.menu)

        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="Carregar Imagem", command=self.load_image)
        file_menu.add_command(label="Salvar Imagem", command=self.save_image)
        self.menu.add_cascade(label="Arquivo", menu=file_menu)

        process_menu = tk.Menu(self.menu, tearoff=0)
        process_menu.add_command(label="Desfazer", command=self.undo)
        process_menu.add_command(label="Histograma", command=self.show_histogram)
        process_menu.add_command(label="Equalizar Histograma", command=self.equalize)
        process_menu.add_command(label="Alargamento de Contraste", command=self.contrast_stretch)
        process_menu.add_command(label="Filtro Média", command=lambda: self.apply_filter('mean'))
        process_menu.add_command(label="Filtro Mediana", command=lambda: self.apply_filter('median'))
        process_menu.add_command(label="Filtro Gaussiano", command=lambda: self.apply_filter('gaussian'))
        process_menu.add_command(label="Filtro Máximo", command=lambda: self.apply_filter('max'))
        process_menu.add_command(label="Filtro Mínimo", command=lambda: self.apply_filter('min'))
        process_menu.add_command(label="Filtro Laplaciano", command=lambda: self.apply_filter('laplacian'))
        process_menu.add_command(label="Filtro Sobel", command=lambda: self.apply_filter('sobel'))
        process_menu.add_command(label="Filtro Prewitt", command=lambda: self.apply_filter('prewitt'))
        process_menu.add_command(label="Filtro Roberts", command=lambda: self.apply_filter('roberts'))
        process_menu.add_command(label="Espectro de Fourier", command=self.show_fourier)
        process_menu.add_command(label="Convolução Fourier (Passa-Alta)", command=lambda: self.apply_fourier_filter('high'))
        process_menu.add_command(label="Convolução Fourier (Passa-Baixa)", command=lambda: self.apply_fourier_filter('low'))
        process_menu.add_command(label="Erosão", command=self.erode)
        process_menu.add_command(label="Dilatação", command=self.dilate)
        process_menu.add_command(label="Segmentação (Otsu)", command=self.otsu)
        self.menu.add_cascade(label="Processamento", menu=process_menu)

    def save_state(self):
        if self.gray_image is not None:
            self.undo_stack.append(self.gray_image.copy())
            # Opcional: limitar o tamanho do histórico
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.gray_image = self.undo_stack.pop()
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)
        else:
            messagebox.showinfo("Desfazer", "Nenhuma ação para desfazer.")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos os arquivos", "*.*")
            ]
        )
        if path:
            self.image = cv2.imread(path)
            if self.image is None:
                messagebox.showerror("Erro", "Erro ao carregar imagem.")
                return
            # Se for RGB, converter para cinza
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                self.gray_image = self.image.copy()
                self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image)
            self.undo_stack.clear()  # Limpa histórico ao carregar nova imagem

    def save_image(self):
        if self.image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.image)

    def display_image(self, img):
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((window_width, window_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

    def show_histogram(self):
        show_histogram(self.gray_image)

    def equalize(self):
        self.save_state()
        self.gray_image = equalize_histogram(self.gray_image)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)

    def contrast_stretch(self):
        self.save_state()
        self.gray_image = contrast_stretching(self.gray_image)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)

    def apply_filter(self, filter_type):
        self.save_state()
        self.gray_image = apply_spatial_filter(self.gray_image, filter_type)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)

    def apply_fourier_filter(self, mode):
        self.save_state()
        self.gray_image = fourier_filter(self.gray_image, mode)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)

    def show_fourier(self):
        show_fourier_spectrum(self.gray_image)

    def erode(self):
        self.save_state()
        self.gray_image = apply_erosion(self.gray_image)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)

    def dilate(self):
        self.save_state()
        self.gray_image = apply_dilation(self.gray_image)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)

    def otsu(self):
        self.save_state()
        self.gray_image = apply_otsu_threshold(self.gray_image)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image)
