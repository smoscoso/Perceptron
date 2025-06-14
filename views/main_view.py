import tkinter as tk
from tkinter import ttk
import time
from utils.ui_components import COLOR_BG, COLOR_PRIMARY, COLOR_TEXT, COLOR_LIGHT_BG
from PIL import Image, ImageTk
import os
import sys

class MainView:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptrón Interactivo - Universidad de Cundinamarca")
        self.root.geometry("1000x750")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(900, 700)
        
        # Crear interfaz principal
        self.create_main_interface()
        
    def create_main_interface(self):
        # Crear marco principal
        self.main_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear encabezado
        self.create_header()
        
        # Crear pestañas
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Pestaña para configuración y entrenamiento
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuración y Entrenamiento")
        
        # Pestaña para visualización de error
        self.error_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.error_frame, text="Error vs Épocas")
        
        # Pestaña para salidas deseadas
        self.desired_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.desired_frame, text="Salidas Deseadas")
        
        # Pestaña para salidas obtenidas
        self.obtained_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.obtained_frame, text="Salidas Obtenidas")
        
        # Pestaña para línea de decisión
        self.decision_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.decision_frame, text="Línea de Decisión")
        
        # Pestaña para pruebas personalizadas
        self.test_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.test_frame, text="Pruebas Personalizadas")
        
        # Crear pie de página
        self.create_footer()

    def obtener_ruta_relativa(self, ruta_archivo):
        if getattr(sys, 'frozen', False):  # Si el programa está empaquetado con PyInstaller
            base_path = sys._MEIPASS       # Carpeta temporal donde PyInstaller extrae archivos
        else:
            base_path = os.path.abspath(".")  # Carpeta normal en modo desarrollo

        return os.path.join(base_path, ruta_archivo)

    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg=COLOR_BG, height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        try:
            # Tamaños deseados
            logo_with = 70   # Ancho
            logo_height = 100  # Alto

            # Crear un frame para contener la imagen
            logo_frame = tk.Frame(header_frame, width=logo_with, height=logo_height, bg=COLOR_BG)
            logo_frame.pack(side=tk.LEFT, padx=15)
            
            try:
                # Obtener la ruta de la imagen de manera segura
                image_path = self.obtener_ruta_relativa(os.path.join("utils", "Images", "escudo_udec.png"))
                
                # Cargar y redimensionar la imagen
                image = Image.open(image_path)
                image = image.resize((logo_with, logo_height), Image.LANCZOS)
                logo_img = ImageTk.PhotoImage(image)

                # Crear un Label con la imagen
                logo_label = tk.Label(logo_frame, image=logo_img, bg=COLOR_BG)
                logo_label.image = logo_img  # Mantener referencia para que no se "pierda" la imagen
                logo_label.pack()

            except Exception as e:
                print(f"Error al cargar la imagen: {e}")
                
                # Como respaldo, dibujamos un canvas con un óvalo verde y texto "UDEC"
                logo_canvas = tk.Canvas(
                    logo_frame, 
                    width=logo_with, 
                    height=logo_height, 
                    bg=COLOR_BG, 
                    highlightthickness=0
                )
                logo_canvas.pack()
                
                logo_canvas.create_oval(
                    5, 5, 
                    logo_with - 5, logo_height - 5, 
                    fill=COLOR_PRIMARY, 
                    outline=""
                )
                logo_canvas.create_text(
                    logo_with / 2, logo_height / 2, 
                    text="UDEC", 
                    fill="white", 
                    font=("Arial", 12, "bold")
                )

        except Exception as e:
            print(f"Error en la creación del logo: {e}")
        # Título y subtítulo
        title_frame = tk.Frame(header_frame, bg=COLOR_BG)
        title_frame.pack(side=tk.LEFT, padx=10)
        
        title_label = tk.Label(title_frame, text="Perceptrón", 
                             font=("Arial", 20, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(title_frame, text="Universidad de Cundinamarca", 
                                font=("Arial", 14), bg=COLOR_BG, fg=COLOR_PRIMARY)
        subtitle_label.pack(anchor='w')
        
        # Información del proyecto
        info_frame = tk.Frame(header_frame, bg=COLOR_BG)
        info_frame.pack(side=tk.RIGHT, padx=15)
        
        info_label = tk.Label(info_frame, text="Compuertas Lógicas", 
                            font=("Arial", 12, "italic"), bg=COLOR_BG, fg=COLOR_TEXT)
        info_label.pack(anchor='e')
        
        date_label = tk.Label(info_frame, text=time.strftime("%d/%m/%Y"), 
                            font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
        date_label.pack(anchor='e')
        
    def create_footer(self):
        footer_frame = tk.Frame(self.main_frame, bg=COLOR_PRIMARY, height=30)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        footer_text = "© Universidad de Cundinamarca - Simulador de Perceptrón para Compuertas Lógicas"
        footer_label = tk.Label(footer_frame, text=footer_text, 
                              font=("Arial", 10), bg=COLOR_PRIMARY, fg="white")
        footer_label.pack(pady=5)
    
    def add_authors_info(self):
        authors_frame = tk.Frame(self.root, bg=COLOR_LIGHT_BG, padx=10, pady=5)
        authors_frame.pack(fill=tk.X, before=self.notebook)
        
        # Separador vertical
        separator = ttk.Separator(authors_frame, orient="vertical")
        separator.pack(side=tk.LEFT, fill="y", padx=10, pady=5)
        
        # Información de los autores
        authors_info = tk.Label(
            authors_frame,
            text="Desarrollado por: Sergio Leonardo Moscoso Ramirez - Zaira Giulianna Salamanca Romero - Miguel Ángel Pardo Lopez",
            font=("Arial", 10, "bold"),
            bg=COLOR_LIGHT_BG,
            fg=COLOR_TEXT
        )
        authors_info.pack(side=tk.LEFT, padx=10)