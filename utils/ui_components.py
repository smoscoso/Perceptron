import tkinter as tk
from tkinter import ttk

# Colores de la Universidad de Cundinamarca
COLOR_PRIMARY = "#004d25"  # Verde oscuro del escudo
COLOR_SECONDARY = "#ffd700"  # Amarillo/dorado del escudo
COLOR_ACCENT_RED = "#e60000"  # Rojo del mapa en el centro
COLOR_ACCENT_BLUE = "#66ccff"  # Azul claro del mapa en el centro
COLOR_BG = "#FFFFFF"  # Fondo blanco para contraste
COLOR_TEXT = "#333333"  # Texto oscuro para mejor legibilidad
COLOR_LIGHT_BG = "#f5f5f5"  # Fondo claro para secciones

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Crear ventana de tooltip
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background=COLOR_PRIMARY, foreground="white",
                         relief="solid", borderwidth=1,
                         font=("Arial", 10, "normal"), padx=5, pady=2)
        label.pack(ipadx=1)
        
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class AnimatedButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        self.hover_bg = kwargs.pop('hover_bg', None)
        self.hover_fg = kwargs.pop('hover_fg', None)
        self.original_bg = kwargs.get('bg', None) or kwargs.get('background', None)
        self.original_fg = kwargs.get('fg', None) or kwargs.get('foreground', None)
        
        tk.Button.__init__(self, master, **kwargs)
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        
    def _on_enter(self, e):
        if self.hover_bg:
            self.config(bg=self.hover_bg)
        if self.hover_fg:
            self.config(fg=self.hover_fg)
            
    def _on_leave(self, e):
        if self.original_bg:
            self.config(bg=self.original_bg)
        if self.original_fg:
            self.config(fg=self.original_fg)
            
    def _on_press(self, e):
        self.config(relief=tk.SUNKEN)
        
    def _on_release(self, e):
        self.config(relief=tk.RAISED)
        self.after(100, lambda: self.config(relief=tk.RAISED))

class ModernFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        self.border_color = kwargs.pop('border_color', COLOR_PRIMARY)
        self.border_width = kwargs.pop('border_width', 2)
        self.corner_radius = kwargs.pop('corner_radius', 10)
        
        tk.Frame.__init__(self, master, **kwargs)
        
        # Crear efecto de borde redondeado
        self.canvas = tk.Canvas(self, bg=self['bg'], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Dibujar borde redondeado
        self.canvas.create_rectangle(
            self.border_width/2, 
            self.border_width/2, 
            self.winfo_reqwidth()-self.border_width/2, 
            self.winfo_reqheight()-self.border_width/2,
            outline=self.border_color, 
            width=self.border_width
        )
        
        # Frame interno para contenido
        self.inner_frame = tk.Frame(self.canvas, bg=self['bg'])
        self.canvas.create_window(
            self.border_width, 
            self.border_width,
            anchor="nw", 
            window=self.inner_frame, 
            width=self.winfo_reqwidth()-2*self.border_width,
            height=self.winfo_reqheight()-2*self.border_width
        )
        
        # Actualizar tamaño del canvas cuando cambia el tamaño del frame
        self.bind("<Configure>", self._on_resize)
        
    def _on_resize(self, event):
        # Actualizar tamaño del canvas y redibujarlo
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            self.border_width/2, 
            self.border_width/2, 
            self.winfo_width()-self.border_width/2, 
            self.winfo_height()-self.border_width/2,
            outline=self.border_color, 
            width=self.border_width
        )
        self.canvas.create_window(
            self.border_width, 
            self.border_width,
            anchor="nw", 
            window=self.inner_frame, 
            width=self.winfo_width()-2*self.border_width,
            height=self.winfo_height()-2*self.border_width
        )

def setup_styles():
    # Configurar estilos para ttk widgets
    style = ttk.Style()
    style.theme_use('default')
    
    # Estilo para pestañas
    style.configure('TNotebook', background=COLOR_BG)
    style.configure('TNotebook.Tab', background=COLOR_PRIMARY, foreground='white', 
                    padding=[15, 5], font=('Arial', 10, 'bold'))
    style.map('TNotebook.Tab', 
              background=[('selected', COLOR_SECONDARY)], 
              foreground=[('selected', COLOR_PRIMARY)])
    
    # Estilo para frames
    style.configure('TFrame', background=COLOR_BG)
    
    # Estilo para combobox
    style.configure('TCombobox', 
                    fieldbackground=COLOR_LIGHT_BG,
                    background=COLOR_PRIMARY,
                    foreground=COLOR_TEXT,
                    arrowcolor=COLOR_PRIMARY)
    
    # Estilo para separadores
    style.configure('TSeparator', background=COLOR_PRIMARY)
    
    # Estilo para barras de progreso
    style.configure("TProgressbar", 
                    troughcolor=COLOR_LIGHT_BG, 
                    background=COLOR_PRIMARY,
                    thickness=10)
    
    return style
def setup_modern_scrollbar_style(style, primary_color, secondary_color, bg_color):
    """
    Configura un estilo moderno para las barras de desplazamiento.
    
    Args:
        style (ttk.Style): El objeto de estilo ttk.
        primary_color (str): Color principal para la barra de desplazamiento.
        secondary_color (str): Color secundario para estados activos.
        bg_color (str): Color de fondo para el canal de la barra.
    """
    style.configure("Modern.Vertical.TScrollbar", 
                   background=primary_color, 
                   troughcolor=bg_color,
                   borderwidth=0,
                   arrowsize=14)
    style.map("Modern.Vertical.TScrollbar",
             background=[('active', secondary_color), ('!active', primary_color)])
    
    style.configure("Modern.Horizontal.TScrollbar", 
                   background=primary_color, 
                   troughcolor=bg_color,
                   borderwidth=0,
                   arrowsize=14)
    style.map("Modern.Horizontal.TScrollbar",
             background=[('active', secondary_color), ('!active', primary_color)])

