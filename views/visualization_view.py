import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.ui_components import COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TEXT, COLOR_BG

class VisualizationView:
    def __init__(self, error_frame, desired_frame, obtained_frame, decision_frame):
        self.error_frame = error_frame
        self.desired_frame = desired_frame
        self.obtained_frame = obtained_frame
        self.decision_frame = decision_frame
        
    def update_error_graph(self, error_history, gate_name, network_type="Perceptron"):
        # Limpiar frame
        for widget in self.error_frame.winfo_children():
            widget.destroy()
            
        # Crear figura
        fig = Figure(figsize=(8, 3), dpi=100)  # Adjusted height for better aspect ratio
        ax = fig.add_subplot(111)
        
        # Graficar error vs épocas
        epochs = range(1, len(error_history) + 1)
        ax.plot(epochs, error_history, color='blue', linewidth=2)
        
        ax.set_title('Error Total vs Épocas', fontsize=12)
        ax.set_xlabel('Época', fontsize=10)
        ax.set_ylabel('Error Total', fontsize=10)
        
        # Configurar grid
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)  # Put grid behind the plot
        
        # Ajustar límites y ticks
        ax.set_ylim(bottom=0)
        
        # Crear canvas
        canvas = FigureCanvasTkAgg(fig, master=self.error_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def update_desired_graph(self, inputs, labels, gate_name):
        # Limpiar frame
        for widget in self.desired_frame.winfo_children():
            widget.destroy()
            
        # Crear figura
        fig = Figure(figsize=(8, 3), dpi=100)  # Adjusted height
        ax = fig.add_subplot(111)
        
        # Preparar datos
        patterns = range(1, len(inputs) + 1)  # Use numerical indices
        
        # Graficar salidas deseadas
        ax.plot(patterns, labels, 'rs-', linewidth=2, markersize=8, 
                label='Salidas Deseadas')
        
        ax.set_title('Salidas Deseadas vs Patrones', fontsize=12)
        ax.set_xlabel('Índice de Patrón', fontsize=10)
        ax.set_ylabel('Salida Deseada', fontsize=10)
        
        # Configurar grid y límites
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0.5, len(patterns) + 0.5)
        
        # Ajustar ticks para mostrar índices enteros
        ax.set_xticks(patterns)
        
        # Crear canvas
        canvas = FigureCanvasTkAgg(fig, master=self.desired_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def update_obtained_graph(self, inputs, predictions, labels, gate_name):
        # Limpiar frame
        for widget in self.obtained_frame.winfo_children():
            widget.destroy()
            
        # Crear figura para salidas obtenidas
        fig_outputs = Figure(figsize=(8, 3), dpi=100)
        ax_outputs = fig_outputs.add_subplot(111)
        
        # Preparar datos
        patterns = range(1, len(inputs) + 1)
        
        # Graficar salidas obtenidas
        ax_outputs.plot(patterns, predictions, 'gd-', linewidth=2, markersize=8,
                       label='Salidas Obtenidas')
        
        ax_outputs.set_title('Salidas Obtenidas vs Patrones', fontsize=12)
        ax_outputs.set_xlabel('Índice de Patrón', fontsize=10)
        ax_outputs.set_ylabel('Salida Obtenida', fontsize=10)
        
        # Configurar grid y límites
        ax_outputs.grid(True, linestyle='-', alpha=0.2)
        ax_outputs.set_axisbelow(True)
        ax_outputs.set_ylim(-0.1, 1.1)
        ax_outputs.set_xlim(0.5, len(patterns) + 0.5)
        
        # Ajustar ticks para mostrar índices enteros
        ax_outputs.set_xticks(patterns)
        
        # Crear canvas
        canvas_outputs = FigureCanvasTkAgg(fig_outputs, master=self.obtained_frame)
        canvas_outputs.draw()
        canvas_outputs.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def update_decision_boundary(self, inputs, labels, gate_name, perceptron=None, 
                                and_perceptron=None, or_perceptron=None, not_perceptron=None):
        # Limpiar frame
        for widget in self.decision_frame.winfo_children():
            widget.destroy()
        
        # Crear un frame principal para contener la gráfica y los pesos
        main_frame = tk.Frame(self.decision_frame, bg=COLOR_BG)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Frame para la gráfica (lado izquierdo)
        graph_frame = tk.Frame(main_frame, bg=COLOR_BG)
        graph_frame.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Frame para los pesos (lado derecho)
        weights_frame = tk.Frame(main_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
        weights_frame.pack(side=tk.RIGHT, fill='y', padx=(10, 0))
        
        # Título para la sección de pesos
        weights_title = tk.Label(weights_frame, text="Pesos del Modelo", 
                               font=("Arial", 12, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
        weights_title.pack(pady=(10, 5), padx=10)
        
        # Crear figura para la línea de decisión
        fig_decision = Figure(figsize=(7, 6), dpi=100)
        ax_decision = fig_decision.add_subplot(111)
        
        # Obtener los puntos para la línea de decisión
        x1_min, x1_max = -0.5, 1.5
        x2_min, x2_max = -0.5, 1.5
        
        # Graficar puntos de clase 0 y 1
        class_0_indices = [i for i, label in enumerate(labels) if label == 0]
        class_1_indices = [i for i, label in enumerate(labels) if label == 1]
        
        class_0 = np.array([inputs[i] for i in class_0_indices])
        class_1 = np.array([inputs[i] for i in class_1_indices])
        
        # Plot class points with better markers and colors
        if len(class_0) > 0:
            ax_decision.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='o', s=100, label='Clase 0', edgecolors='black', linewidth=1.5)
        if len(class_1) > 0:
            ax_decision.scatter(class_1[:, 0], class_1[:, 1], color='blue', marker='^', s=100, label='Clase 1', edgecolors='black', linewidth=1.5)
        
        # Special case for XOR with three perceptrons
        if gate_name == "XOR" and and_perceptron is not None and or_perceptron is not None and not_perceptron is not None:
            # Create a grid of points for the decision boundary
            x1_grid = np.linspace(x1_min, x1_max, 100)
            x2_grid = np.linspace(x2_min, x2_max, 100)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            
            # Initialize grid for predictions
            grid_shape = X1.shape
            Z = np.zeros(grid_shape)
            
            # Fill the grid with predictions
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    input_point = np.array([X1[i, j], X2[i, j]])
                    or_output = or_perceptron.predict(input_point)
                    and_output = and_perceptron.predict(input_point)
                    not_and_output = not_perceptron.predict(np.array([and_output]))
                    Z[i, j] = 1 if or_output == 1 and not_and_output == 1 else 0
            
            # Plot the decision boundary
            ax_decision.contourf(X1, X2, Z, alpha=0.3, cmap='coolwarm')
            
            # Draw the AND perceptron line
            w_and = and_perceptron.weights
            if w_and[1] != 0:  # Avoid division by zero
                x1_line = np.array([x1_min, x1_max])
                x2_line = (-w_and[2] - w_and[0] * x1_line) / w_and[1]
                ax_decision.plot(x1_line, x2_line, 'g-', linewidth=2, label='Línea AND')
            
            # Draw the OR perceptron line
            w_or = or_perceptron.weights
            if w_or[1] != 0:  # Avoid division by zero
                x1_line = np.array([x1_min, x1_max])
                x2_line = (-w_or[2] - w_or[0] * x1_line) / w_or[1]
                ax_decision.plot(x1_line, x2_line, 'm-', linewidth=2, label='Línea OR')
            
            # Add explanation text
            ax_decision.text(0.5, -0.1, "XOR implementado con tres perceptrones: AND, OR y NOT", 
                          transform=ax_decision.transAxes, ha='center', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Mostrar pesos en el panel lateral
            w_and = and_perceptron.weights
            w_or = or_perceptron.weights
            w_not = not_perceptron.weights
            
            # Frame para perceptrón AND
            and_frame = tk.Frame(weights_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
            and_frame.pack(fill='x', padx=10, pady=5)
            
            and_title = tk.Label(and_frame, text="Perceptrón AND", 
                               font=("Arial", 11, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
            and_title.pack(pady=(5, 2))
            
            and_w0 = tk.Label(and_frame, text=f"w0 = {w_and[0]}", 
                            font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            and_w0.pack(anchor='w', padx=10)
            
            and_w1 = tk.Label(and_frame, text=f"w1 = {w_and[1]}", 
                            font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            and_w1.pack(anchor='w', padx=10)
            
            and_w2 = tk.Label(and_frame, text=f"w2 = {w_and[2]}", 
                            font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            and_w2.pack(anchor='w', padx=10)
            
            # Frame para perceptrón OR
            or_frame = tk.Frame(weights_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
            or_frame.pack(fill='x', padx=10, pady=5)
            
            or_title = tk.Label(or_frame, text="Perceptrón OR", 
                              font=("Arial", 11, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
            or_title.pack(pady=(5, 2))
            
            or_w0 = tk.Label(or_frame, text=f"w0 = {w_or[0]}", 
                           font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            or_w0.pack(anchor='w', padx=10)
            
            or_w1 = tk.Label(or_frame, text=f"w1 = {w_or[1]}", 
                           font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            or_w1.pack(anchor='w', padx=10)
            
            or_w2 = tk.Label(or_frame, text=f"w2 = {w_or[2]}", 
                           font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            or_w2.pack(anchor='w', padx=10)
            
            # Frame para perceptrón NOT
            not_frame = tk.Frame(weights_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
            not_frame.pack(fill='x', padx=10, pady=5)
            
            not_title = tk.Label(not_frame, text="Perceptrón NOT", 
                               font=("Arial", 11, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
            not_title.pack(pady=(5, 2))
            
            not_w0 = tk.Label(not_frame, text=f"w0 = {w_not[0]}", 
                            font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            not_w0.pack(anchor='w', padx=10)
            
            not_w1 = tk.Label(not_frame, text=f"w1 = {w_not[1]}", 
                            font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            not_w1.pack(anchor='w', padx=10)
            
        # Regular perceptron case
        elif perceptron is not None:
            # Obtener pesos del perceptrón
            w = perceptron.weights
            
            # Calcular puntos para la línea de decisión: w0*x0 + w1*x1 + w2 = 0
            # Despejando: x1 = (-w2 - w0*x0) / w1
            if w[1] != 0:  # Evitar división por cero
                x1 = np.array([x1_min, x1_max])
                x2 = (-w[2] - w[0] * x1) / w[1]
                ax_decision.plot(x1, x2, 'k-', linewidth=2, label='Línea de Decisión')
            else:
                # Línea vertical si w1 es cero
                ax_decision.axvline(x=-w[2]/w[0], color='k', linewidth=2, label='Línea de Decisión')
            
            # Mostrar pesos en el panel lateral
            perceptron_frame = tk.Frame(weights_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
            perceptron_frame.pack(fill='x', padx=10, pady=5)
            
            perceptron_title = tk.Label(perceptron_frame, text=f"Perceptrón {gate_name}", 
                                      font=("Arial", 11, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
            perceptron_title.pack(pady=(5, 2))
            
            perceptron_w0 = tk.Label(perceptron_frame, text=f"w0 = {w[0]}", 
                                   font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            perceptron_w0.pack(anchor='w', padx=10)
            
            perceptron_w1 = tk.Label(perceptron_frame, text=f"w1 = {w[1]}", 
                                   font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            perceptron_w1.pack(anchor='w', padx=10)
            
            perceptron_w2 = tk.Label(perceptron_frame, text=f"w2 = {w[2]}", 
                                   font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            perceptron_w2.pack(anchor='w', padx=10)
            
            # Ecuación de la línea de decisión
            equation_frame = tk.Frame(weights_frame, bg=COLOR_BG)
            equation_frame.pack(fill='x', padx=10, pady=10)
            
            equation_title = tk.Label(equation_frame, text="Ecuación de la línea de decisión:", 
                                    font=("Arial", 10, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
            equation_title.pack(anchor='w')
            
            if w[1] != 0:
                equation_text = f"{w[0]}*x1 + {w[1]}*x2 + {w[2]} = 0"
            else:
                equation_text = f"{w[0]}*x1 + {w[2]} = 0"
                
            equation_label = tk.Label(equation_frame, text=equation_text, 
                                    font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT)
            equation_label.pack(anchor='w', padx=10)
            
        else:
            # Si no hay perceptrón, mostrar mensaje informativo para XOR
            if gate_name == "XOR":
                info_text = ax_decision.text(0.5, 0.5, 
                                          "La compuerta XOR no es linealmente separable.\nSe requieren tres perceptrones para su implementación.",
                                          ha='center', va='center', fontsize=12, transform=ax_decision.transAxes,
                                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
                # Mensaje en el panel lateral
                info_frame = tk.Frame(weights_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
                info_frame.pack(fill='x', padx=10, pady=5)
                
                info_label = tk.Label(info_frame, text="XOR no es linealmente separable", 
                                    font=("Arial", 11, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
                info_label.pack(pady=5)
                
                info_desc = tk.Label(info_frame, text="Se requieren múltiples perceptrones\npara implementar esta compuerta.", 
                                   font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT, justify=tk.LEFT)
                info_desc.pack(pady=5)
                
            else:
                # Si no hay perceptrón, usar una línea genérica según la compuerta
                x1 = np.array([x1_min, x1_max])
                if gate_name == "OR" or gate_name == "NAND":
                    x2 = -x1 + 0.5  # Línea diagonal para OR/NAND
                else:  # AND o NOR
                    x2 = -x1 + 1.5  # Línea diagonal para AND/NOR
                ax_decision.plot(x1, x2, 'k-', linewidth=2, label='Línea de Decisión (Aproximada)')
                
                # Mensaje en el panel lateral
                info_frame = tk.Frame(weights_frame, bg=COLOR_BG, bd=1, relief=tk.GROOVE)
                info_frame.pack(fill='x', padx=10, pady=5)
                
                info_label = tk.Label(info_frame, text="Perceptrón no entrenado", 
                                    font=("Arial", 11, "bold"), bg=COLOR_BG, fg=COLOR_PRIMARY)
                info_label.pack(pady=5)
                
                info_desc = tk.Label(info_frame, text="Entrene el perceptrón para\nver los pesos reales.", 
                                   font=("Arial", 10), bg=COLOR_BG, fg=COLOR_TEXT, justify=tk.LEFT)
                info_desc.pack(pady=5)
        
        ax_decision.set_title(f'Perceptrón para {gate_name} - Línea de Decisión', fontsize=12)
        ax_decision.set_xlabel('x1', fontsize=10)
        ax_decision.set_ylabel('x2', fontsize=10)
        ax_decision.grid(True, linestyle='-', alpha=0.2)
        ax_decision.set_axisbelow(True)
        
        # Ajustar límites
        ax_decision.set_xlim(x1_min, x1_max)
        ax_decision.set_ylim(x2_min, x2_max)
        
        # Añadir leyenda
        ax_decision.legend(loc='upper right')
        
        # Crear canvas
        canvas_decision = FigureCanvasTkAgg(fig_decision, master=graph_frame)
        canvas_decision.draw()
        canvas_decision.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)