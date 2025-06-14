import tkinter as tk
from tkinter import ttk
from utils.ui_components import (
    COLOR_BG, COLOR_LIGHT_BG, COLOR_PRIMARY, COLOR_SECONDARY, 
    COLOR_TEXT, COLOR_ACCENT_RED, ToolTip, AnimatedButton
)

class ConfigView:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.setup_config_tab()
        
    def setup_config_tab(self):
        # Frame principal para la configuración
        main_config = tk.Frame(self.parent, bg=COLOR_BG)
        main_config.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Dividir en dos columnas
        left_column = tk.Frame(main_config, bg=COLOR_BG)
        left_column.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 10))
        
        right_column = tk.Frame(main_config, bg=COLOR_BG)
        right_column.pack(side=tk.RIGHT, fill='both', expand=True, padx=(10, 0))
        
        # Panel de configuración (izquierda)
        config_panel = tk.Frame(left_column, bg=COLOR_LIGHT_BG, bd=1, relief=tk.GROOVE)
        config_panel.pack(fill='both', expand=True, pady=(0, 0))
        
        config_title = tk.Label(config_panel, text="Configuración del Perceptrón", 
                              font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        config_title.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Separador
        separator = ttk.Separator(config_panel, orient='horizontal')
        separator.pack(fill='x', padx=15, pady=5)
        
        # Frame para controles
        controls = tk.Frame(config_panel, bg=COLOR_LIGHT_BG, padx=20, pady=10)
        controls.pack(fill='x')
        
        # Selección de compuerta lógica
        gate_frame = tk.Frame(controls, bg=COLOR_LIGHT_BG)
        gate_frame.pack(fill='x', pady=10)
        
        gate_label = tk.Label(gate_frame, text="Compuerta Lógica:", 
                            font=("Arial", 12, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        gate_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.gate_var = tk.StringVar(value="AND")
        self.gate_combo = ttk.Combobox(gate_frame, textvariable=self.gate_var, 
                                     width=15, font=("Arial", 12), state="readonly")
        self.gate_combo.pack(side=tk.LEFT)
        
        ToolTip(self.gate_combo, "Seleccione la compuerta lógica a simular")
        
        # Parámetros de entrenamiento - Solo épocas ya que la tasa de aprendizaje es interna
        params_frame = tk.Frame(controls, bg=COLOR_LIGHT_BG)
        params_frame.pack(fill='x', pady=10)
        
        # Épocas 
        epochs_label = tk.Label(params_frame, text="Épocas máximas:", 
                              font=("Arial", 12), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        epochs_label.grid(row=0, column=0, sticky='w', padx=(0, 10), pady=5)
        
        self.epochs_var = tk.IntVar(value=1000000)  # Sin límite real
        epochs_entry = tk.Entry(params_frame, textvariable=self.epochs_var, width=10, 
                              font=("Arial", 12), bg=COLOR_LIGHT_BG, state='disabled')
        epochs_entry.grid(row=0, column=1, sticky='w', pady=5)
        
        note_label = tk.Label(params_frame, text="(Ilimitadas)", 
                           font=("Arial", 10, "italic"), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        note_label.grid(row=0, column=2, sticky='w', padx=5, pady=5)
        
        ToolTip(epochs_entry, "El entrenamiento continuará hasta encontrar una solución")
        
        # Frame para botones de entrenamiento y edición de pesos
        buttons_frame = tk.Frame(config_panel, bg=COLOR_LIGHT_BG)
        buttons_frame.pack(fill='x', pady=15)

        # Frame interno para organizar los botones en una fila
        buttons_row = tk.Frame(buttons_frame, bg=COLOR_LIGHT_BG)
        buttons_row.pack(pady=10)

        # Botón de entrenamiento (mitad del ancho)
        self.train_button = AnimatedButton(
            buttons_row, text="Entrenar Perceptrón", 
            font=("Arial", 12, "bold"), 
            bg=COLOR_PRIMARY, fg="white",
            hover_bg=COLOR_SECONDARY, hover_fg=COLOR_PRIMARY,
            activebackground=COLOR_SECONDARY, activeforeground=COLOR_PRIMARY,
            padx=10, pady=10, relief=tk.RAISED, bd=2,
            width=15
        )
        self.train_button.pack(side=tk.LEFT, padx=(0, 5))

        # Botón para reentrenar la red (mitad del ancho)
        self.apply_weights_button = AnimatedButton(
            buttons_row, text="Validar Pesos", 
            font=("Arial", 12, "bold"), 
            bg=COLOR_PRIMARY, fg="white",
            hover_bg=COLOR_SECONDARY, hover_fg=COLOR_PRIMARY,
            activebackground=COLOR_SECONDARY, activeforeground=COLOR_PRIMARY,
            padx=10, pady=10, relief=tk.RAISED, bd=2,
            width=15,
            state=tk.DISABLED  # Inicialmente deshabilitado
        )
        self.apply_weights_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(config_panel, orient="horizontal", 
                                      length=300, mode="determinate", style="TProgressbar")
        self.progress.pack(pady=(0, 15), padx=20)
        
        # Estado del entrenamiento (ahora debajo del botón)
        self.status_frame = tk.Frame(config_panel, bg=COLOR_LIGHT_BG)
        self.status_frame.pack(fill='x', pady=(0, 15))
        
        # Contenedor centrado para el estado
        status_center_frame = tk.Frame(self.status_frame, bg=COLOR_LIGHT_BG)
        status_center_frame.pack(anchor='center')
        
        self.config_status_indicator = tk.Canvas(status_center_frame, width=20, height=20, 
                                        bg=COLOR_LIGHT_BG, highlightthickness=0)
        self.config_status_indicator.pack(side=tk.LEFT, padx=5)
        self.config_status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_ACCENT_RED, outline="")
        
        self.config_status_label = tk.Label(status_center_frame, text="Estado: No entrenado", 
                                   font=("Arial", 12, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_ACCENT_RED)
        self.config_status_label.pack(side=tk.LEFT)
        
        # Sección para editar pesos (ahora justo debajo del estado)
        self.edit_weights_frame = tk.Frame(config_panel, bg=COLOR_LIGHT_BG, bd=1, relief=tk.GROOVE)
        self.edit_weights_frame.pack(fill='x', pady=5, padx=20)

        edit_weights_title = tk.Label(self.edit_weights_frame, text="Editar Pesos:", 
                                    font=("Arial", 12, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        edit_weights_title.pack(anchor='w', padx=5, pady=5)

        # Crear un frame con scrollbar para los pesos
        weights_scroll_frame = tk.Frame(self.edit_weights_frame, bg=COLOR_LIGHT_BG)
        weights_scroll_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Canvas y scrollbar modernos para hacer desplazable
        weights_canvas = tk.Canvas(weights_scroll_frame, bg=COLOR_LIGHT_BG, highlightthickness=0)
        weights_scrollbar = ttk.Scrollbar(weights_scroll_frame, orient="vertical", command=weights_canvas.yview, style="Modern.Vertical.TScrollbar")

        # Configurar el canvas
        weights_canvas.configure(yscrollcommand=weights_scrollbar.set)
        weights_canvas.pack(side=tk.LEFT, fill='both', expand=True)
        weights_scrollbar.pack(side=tk.RIGHT, fill='y')

        # Frame interior para los controles de pesos
        weights_inner_frame = tk.Frame(weights_canvas, bg=COLOR_LIGHT_BG)
        weights_canvas.create_window((0, 0), window=weights_inner_frame, anchor='nw', tags="weights_inner_frame")

        # Configurar evento para ajustar el tamaño del scrollregion
        def _configure_weights_canvas(event):
            weights_canvas.configure(scrollregion=weights_canvas.bbox("all"))
            weights_canvas.itemconfig("weights_inner_frame", width=weights_canvas.winfo_width())

        weights_inner_frame.bind("<Configure>", _configure_weights_canvas)
        weights_canvas.bind("<Configure>", lambda e: weights_canvas.itemconfig("weights_inner_frame", width=e.width))

        # Frame para perceptrón simple
        self.simple_edit_frame = tk.Frame(weights_inner_frame, bg=COLOR_LIGHT_BG)
        self.simple_edit_frame.pack(fill='x', padx=10, pady=5)

        # Campos para editar pesos del perceptrón simple
        simple_w0_label = tk.Label(self.simple_edit_frame, text="w0:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        simple_w0_label.grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.simple_w0_var = tk.StringVar()
        simple_w0_entry = tk.Entry(self.simple_edit_frame, textvariable=self.simple_w0_var, width=10)
        simple_w0_entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')

        simple_w1_label = tk.Label(self.simple_edit_frame, text="w1:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        simple_w1_label.grid(row=0, column=2, padx=5, pady=2, sticky='e')
        self.simple_w1_var = tk.StringVar()
        simple_w1_entry = tk.Entry(self.simple_edit_frame, textvariable=self.simple_w1_var, width=10)
        simple_w1_entry.grid(row=0, column=3, padx=5, pady=2, sticky='w')

        simple_w2_label = tk.Label(self.simple_edit_frame, text="w2:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        simple_w2_label.grid(row=0, column=4, padx=5, pady=2, sticky='e')
        self.simple_w2_var = tk.StringVar()
        simple_w2_entry = tk.Entry(self.simple_edit_frame, textvariable=self.simple_w2_var, width=10)
        simple_w2_entry.grid(row=0, column=5, padx=5, pady=2, sticky='w')

        # Frame para perceptrones múltiples (XOR)
        self.xor_edit_frame = tk.Frame(weights_inner_frame, bg=COLOR_LIGHT_BG)

        # AND perceptron
        and_label = tk.Label(self.xor_edit_frame, text="AND:", font=("Arial", 10, "bold"), bg=COLOR_LIGHT_BG)
        and_label.grid(row=0, column=0, padx=5, pady=2, sticky='e')

        and_w0_label = tk.Label(self.xor_edit_frame, text="w0:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        and_w0_label.grid(row=0, column=1, padx=5, pady=2, sticky='e')
        self.and_w0_var = tk.StringVar()
        and_w0_entry = tk.Entry(self.xor_edit_frame, textvariable=self.and_w0_var, width=10)
        and_w0_entry.grid(row=0, column=2, padx=5, pady=2, sticky='w')

        and_w1_label = tk.Label(self.xor_edit_frame, text="w1:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        and_w1_label.grid(row=0, column=3, padx=5, pady=2, sticky='e')
        self.and_w1_var = tk.StringVar()
        and_w1_entry = tk.Entry(self.xor_edit_frame, textvariable=self.and_w1_var, width=10)
        and_w1_entry.grid(row=0, column=4, padx=5, pady=2, sticky='w')

        and_w2_label = tk.Label(self.xor_edit_frame, text="w2:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        and_w2_label.grid(row=0, column=5, padx=5, pady=2, sticky='e')
        self.and_w2_var = tk.StringVar()
        and_w2_entry = tk.Entry(self.xor_edit_frame, textvariable=self.and_w2_var, width=10)
        and_w2_entry.grid(row=0, column=6, padx=5, pady=2, sticky='w')

        # OR perceptron
        or_label = tk.Label(self.xor_edit_frame, text="OR:", font=("Arial", 10, "bold"), bg=COLOR_LIGHT_BG)
        or_label.grid(row=1, column=0, padx=5, pady=2, sticky='e')

        or_w0_label = tk.Label(self.xor_edit_frame, text="w0:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        or_w0_label.grid(row=1, column=1, padx=5, pady=2, sticky='e')
        self.or_w0_var = tk.StringVar()
        or_w0_entry = tk.Entry(self.xor_edit_frame, textvariable=self.or_w0_var, width=10)
        or_w0_entry.grid(row=1, column=2, padx=5, pady=2, sticky='w')

        or_w1_label = tk.Label(self.xor_edit_frame, text="w1:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        or_w1_label.grid(row=1, column=3, padx=5, pady=2, sticky='e')
        self.or_w1_var = tk.StringVar()
        or_w1_entry = tk.Entry(self.xor_edit_frame, textvariable=self.or_w1_var, width=10)
        or_w1_entry.grid(row=1, column=4, padx=5, pady=2, sticky='w')

        or_w2_label = tk.Label(self.xor_edit_frame, text="w2:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        or_w2_label.grid(row=1, column=5, padx=5, pady=2, sticky='e')
        self.or_w2_var = tk.StringVar()
        or_w2_entry = tk.Entry(self.xor_edit_frame, textvariable=self.or_w2_var, width=10)
        or_w2_entry.grid(row=1, column=6, padx=5, pady=2, sticky='w')

        # NOT perceptron
        not_label = tk.Label(self.xor_edit_frame, text="NOT:", font=("Arial", 10, "bold"), bg=COLOR_LIGHT_BG)
        not_label.grid(row=2, column=0, padx=5, pady=2, sticky='e')

        not_w0_label = tk.Label(self.xor_edit_frame, text="w0:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        not_w0_label.grid(row=2, column=1, padx=5, pady=2, sticky='e')
        self.not_w0_var = tk.StringVar()
        not_w0_entry = tk.Entry(self.xor_edit_frame, textvariable=self.not_w0_var, width=10)
        not_w0_entry.grid(row=2, column=2, padx=5, pady=2, sticky='w')

        not_w1_label = tk.Label(self.xor_edit_frame, text="w1:", font=("Arial", 10), bg=COLOR_LIGHT_BG)
        not_w1_label.grid(row=2, column=3, padx=5, pady=2, sticky='e')
        self.not_w1_var = tk.StringVar()
        not_w1_entry = tk.Entry(self.xor_edit_frame, textvariable=self.not_w1_var, width=10)
        not_w1_entry.grid(row=2, column=4, padx=5, pady=2, sticky='w')

        # Ocultar frames de edición inicialmente
        self.simple_edit_frame.pack_forget()
        self.xor_edit_frame.pack_forget()
        
        # Panel de tabla de verdad (derecha arriba)
        truth_panel = tk.Frame(right_column, bg=COLOR_LIGHT_BG, bd=1, relief=tk.GROOVE)
        truth_panel.pack(fill='both', expand=True, pady=(0, 10))
        
        truth_title = tk.Label(truth_panel, text="Tabla de Verdad", 
                             font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        truth_title.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Separador
        separator2 = ttk.Separator(truth_panel, orient='horizontal')
        separator2.pack(fill='x', padx=15, pady=5)
        
        # Tabla de verdad
        table_frame = tk.Frame(truth_panel, bg=COLOR_LIGHT_BG)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Encabezados
        header_frame = tk.Frame(table_frame, bg=COLOR_PRIMARY)
        header_frame.pack(fill='x', pady=(0, 2))
        
        headers = ["Entrada 1", "Entrada 2", "Salida Esperada"]
        for i, header in enumerate(headers):
            header_label = tk.Label(header_frame, text=header, font=("Arial", 12, "bold"), 
                                  bg=COLOR_PRIMARY, fg="white", width=12, padx=5, pady=5)
            header_label.grid(row=0, column=i, sticky='nsew')
            header_frame.grid_columnconfigure(i, weight=1)
        
        # Contenido de la tabla
        self.table_content = tk.Frame(table_frame, bg=COLOR_LIGHT_BG)
        self.table_content.pack(fill='both', expand=True)
        
        # Panel de resultados (derecha abajo)
        results_panel = tk.Frame(right_column, bg=COLOR_LIGHT_BG, bd=1, relief=tk.GROOVE)
        results_panel.pack(fill='both', expand=True)
        
        results_title = tk.Label(results_panel, text="Resultados del Entrenamiento", 
                               font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        results_title.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Separador
        separator3 = ttk.Separator(results_panel, orient='horizontal')
        separator3.pack(fill='x', padx=15, pady=5)
        
        # Crear un frame con scrollbar para los resultados
        results_scroll_frame = tk.Frame(results_panel, bg=COLOR_LIGHT_BG)
        results_scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas y scrollbar para hacer desplazable
        results_canvas = tk.Canvas(results_scroll_frame, bg=COLOR_LIGHT_BG, highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(results_scroll_frame, orient="vertical", command=results_canvas.yview, style="Modern.Vertical.TScrollbar")
        
        # Configurar el canvas
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        results_canvas.pack(side=tk.LEFT, fill='both', expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Frame interior para los resultados
        self.results_frame = tk.Frame(results_canvas, bg=COLOR_LIGHT_BG)
        results_canvas.create_window((0, 0), window=self.results_frame, anchor='nw', tags="self.results_frame")
        
        # Configurar evento para ajustar el tamaño del scrollregion
        def _configure_results_canvas(event):
            results_canvas.configure(scrollregion=results_canvas.bbox("all"))
            results_canvas.itemconfig("self.results_frame", width=results_canvas.winfo_width())
        
        self.results_frame.bind("<Configure>", _configure_results_canvas)
        results_canvas.bind("<Configure>", lambda e: results_canvas.itemconfig("self.results_frame", width=e.width))
        
        # Etiquetas para resultados
        self.epochs_result = tk.Label(self.results_frame, text="Épocas utilizadas: 0", 
                                    font=("Arial", 12), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        self.epochs_result.pack(anchor='w', pady=2)
        
        self.accuracy_result = tk.Label(self.results_frame, text="Precisión: 0%", 
                                      font=("Arial", 12), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        self.accuracy_result.pack(anchor='w', pady=2)
        
        # Crear un frame para los pesos
        self.weights_frame = tk.Frame(self.results_frame, bg=COLOR_LIGHT_BG)
        self.weights_frame.pack(fill='x', pady=2)
        
        self.weights_title = tk.Label(self.weights_frame, text="Pesos finales:", 
                                    font=("Arial", 12, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        self.weights_title.pack(anchor='w')
        
        # Crear un frame para mostrar los pesos detallados
        self.weights_detail_frame = tk.Frame(self.weights_frame, bg=COLOR_LIGHT_BG, padx=15)
        self.weights_detail_frame.pack(fill='x', pady=5)
        
        # Etiqueta para pesos de perceptrón simple
        self.weights_simple = tk.Label(self.weights_detail_frame, text="", 
                                     font=("Arial", 11), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        self.weights_simple.pack(anchor='w', pady=1)
        
        # Etiquetas para pesos de perceptrones múltiples
        self.weights_and = tk.Label(self.weights_detail_frame, text="", 
                                  font=("Arial", 11), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        
        self.weights_or = tk.Label(self.weights_detail_frame, text="", 
                                 font=("Arial", 11), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        
        self.weights_not = tk.Label(self.weights_detail_frame, text="", 
                                  font=("Arial", 11), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        
        self.error_result = tk.Label(self.results_frame, text="Error final: 0", 
                                   font=("Arial", 12), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        self.error_result.pack(anchor='w', pady=2)
        
        # Estado del entrenamiento
        self.status_frame = tk.Frame(self.results_frame, bg=COLOR_LIGHT_BG)
        self.status_frame.pack(fill='x', pady=10)
        
        self.status_label = tk.Label(self.status_frame, text="Estado: No entrenado", 
                                   font=("Arial", 12, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_ACCENT_RED)
        self.status_label.pack(side=tk.LEFT)
        
        # Indicador visual
        self.status_indicator = tk.Canvas(self.status_frame, width=20, height=20, 
                                        bg=COLOR_LIGHT_BG, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=10)
        self.status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_ACCENT_RED, outline="")
        
    def update_truth_table(self, inputs, labels):
        # Limpiar tabla actual
        for widget in self.table_content.winfo_children():
            widget.destroy()
            
        # Llenar tabla
        for i, (input_row, label) in enumerate(zip(inputs, labels)):
            row_bg = COLOR_LIGHT_BG if i % 2 == 0 else "white"
            row_frame = tk.Frame(self.table_content, bg=row_bg)
            row_frame.pack(fill='x')
            
            # Entrada 1
            input1_label = tk.Label(row_frame, text=str(input_row[0]), font=("Arial", 12), 
                                  bg=row_bg, fg=COLOR_TEXT, width=12, padx=5, pady=8)
            input1_label.grid(row=0, column=0, sticky='nsew')
            
            # Entrada 2
            input2_label = tk.Label(row_frame, text=str(input_row[1]), font=("Arial", 12), 
                                  bg=row_bg, fg=COLOR_TEXT, width=12, padx=5, pady=8)
            input2_label.grid(row=0, column=1, sticky='nsew')
            
            # Salida esperada
            output_label = tk.Label(row_frame, text=str(label), font=("Arial", 12, "bold"), 
                                  bg=row_bg, fg=COLOR_PRIMARY, width=12, padx=5, pady=8)
            output_label.grid(row=0, column=2, sticky='nsew')
            
            # Configurar columnas
            for j in range(3):
                row_frame.grid_columnconfigure(j, weight=1)
                
    # Añadir método para actualizar los resultados con los pesos completos

    def update_results(self, epochs, accuracy, weights_info, error_final, success, perceptron=None, three_perceptron_xor=None):
        # Actualizar resultados
        self.epochs_result.config(text=f"Épocas utilizadas: {epochs}")
        self.accuracy_result.config(text=f"Precisión: {accuracy:.2f}%")
        
        # Limpiar etiquetas de pesos anteriores
        self.weights_simple.pack_forget()
        self.weights_and.pack_forget()
        self.weights_or.pack_forget()
        self.weights_not.pack_forget()
        
        # Mostrar pesos según el tipo de modelo
        if three_perceptron_xor is not None:
            # Mostrar pesos para el modelo de tres perceptrones
            and_weights = three_perceptron_xor.and_perceptron.weights
            or_weights = three_perceptron_xor.or_perceptron.weights
            not_weights = three_perceptron_xor.not_perceptron.weights
            
            self.weights_and.config(text=f"AND: w0={and_weights[0]}, w1={and_weights[1]}, w2={and_weights[2]}")
            self.weights_or.config(text=f"OR: w0={or_weights[0]}, w1={or_weights[1]}, w2={or_weights[2]}")
            self.weights_not.config(text=f"NOT: w0={not_weights[0]}, w1={not_weights[1]}")
            
            self.weights_and.pack(anchor='w', pady=1)
            self.weights_or.pack(anchor='w', pady=1)
            self.weights_not.pack(anchor='w', pady=1)
        elif perceptron is not None:
            # Mostrar pesos para el perceptrón simple
            weights = perceptron.weights
            self.weights_simple.config(text=f"w0={weights[0]}, w1={weights[1]}, w2={weights[2]}")
            self.weights_simple.pack(anchor='w', pady=1)
        
        self.error_result.config(text=f"Error final: {error_final}")
        
        # Actualizar ambos indicadores de estado
        if success:
            # Estado en el panel de resultados
            self.status_label.config(text="Estado: Entrenamiento exitoso", fg=COLOR_PRIMARY)
            self.status_indicator.delete("all")
            self.status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_PRIMARY, outline="")
            
            # Estado en el panel de configuración (debajo de los botones)
            self.config_status_label.config(text="Estado: Entrenamiento exitoso", fg=COLOR_PRIMARY)
            self.config_status_indicator.delete("all")
            self.config_status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_PRIMARY, outline="")
        
            # Habilitar botón de reentrenamiento
            self.apply_weights_button.config(state=tk.NORMAL)
        else:
            # Estado en el panel de resultados
            self.status_label.config(text="Estado: Entrenamiento incompleto", fg=COLOR_ACCENT_RED)
            self.status_indicator.delete("all")
            self.status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_ACCENT_RED, outline="")
            
            # Estado en el panel de configuración (debajo de los botones)
            self.config_status_label.config(text="Estado: Entrenamiento incompleto", fg=COLOR_ACCENT_RED)
            self.config_status_indicator.delete("all")
            self.config_status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_ACCENT_RED, outline="")
        
            # Deshabilitar botón de reentrenamiento
            self.apply_weights_button.config(state=tk.DISABLED)