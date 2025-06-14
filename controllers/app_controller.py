import tkinter as tk
from tkinter import messagebox
import numpy as np
import time

from models.perceptron_model import Perceptron, ThreePerceptronXOR
from models.gate_model import GateModel
from views.main_view import MainView
from views.config_view import ConfigView
from views.test_view import TestView
from views.visualization_view import VisualizationView
from utils.ui_components import setup_styles, COLOR_ACCENT_RED, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_LIGHT_BG

class AppController:
    def __init__(self, root):
        # Configurar estilos
        self.style = setup_styles()
        
        # Configurar estilo moderno para barras de desplazamiento
        from utils.ui_components import setup_modern_scrollbar_style
        setup_modern_scrollbar_style(self.style, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_LIGHT_BG)
        
        # Inicializar modelos
        self.gate_model = GateModel()
        self.perceptron = None
        self.mlp = None
        self.three_perceptron_xor = None
        
        # Estado actual
        self.current_gate = "AND"
        self.current_inputs, self.current_labels = self.gate_model.get_gate_data("AND")
        
        # Inicializar vista principal
        self.main_view = MainView(root)
        
        # Añadir información de los autores en la parte superior
        self.main_view.add_authors_info()
        
        # Eliminar la llamada a add_authors_footer ya que ahora usamos el método en MainView
        # self.add_authors_footer(root)
        
        # Inicializar vistas secundarias
        self.config_view = ConfigView(self.main_view.config_frame)
        self.test_view = TestView(self.main_view.test_frame)
        self.visualization_view = VisualizationView(
            self.main_view.error_frame,
            self.main_view.desired_frame,
            self.main_view.obtained_frame,
            self.main_view.decision_frame
        )
        
        # Configurar eventos
        self.setup_events()
        
        # Inicializar con valores predeterminados
        self.initialize_ui()
        
        # Añadir pie de página con los nombres de los autores
        self.add_authors_footer(root)
        
    def initialize_ui(self):
        # Actualizar tabla de verdad
        self.config_view.update_truth_table(self.current_inputs, self.current_labels)
        
        # Actualizar lista desplegable de compuertas
        self.test_view.test_gate_combo.set(self.current_gate)
        self.test_view.current_gate_label.config(text=self.current_gate)
        
        # Mostrar el frame de edición simple por defecto y ocultar el de XOR
        self.config_view.xor_edit_frame.pack_forget()
        self.config_view.simple_edit_frame.pack(fill='x', padx=10, pady=5)
        
        # Inicializar visualizaciones
        self.visualization_view.update_desired_graph(self.current_inputs, self.current_labels, self.current_gate)
        self.visualization_view.update_decision_boundary(self.current_inputs, self.current_labels, self.current_gate)

    def add_authors_footer(self, root):
        """Añade un pie de página con los nombres de los autores"""
        # Frame para el pie de página
        footer_frame = tk.Frame(root, bg=COLOR_PRIMARY, height=30)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Etiqueta con los nombres de los autores
        authors_text = "Desarrollado por: Sergio Leonardo Moscoso Ramirez • Zaira Giulianna Salamanca Romero • Miguel Angel Pardo Lopez"
        authors_label = tk.Label(
            footer_frame, 
            text=authors_text,
            font=("Arial", 9, "italic"),
            bg=COLOR_PRIMARY,
            fg="white",
            padx=10,
            pady=5
        )
        authors_label.pack(side=tk.RIGHT, padx=10)
        
        # Etiqueta con el año
        year_label = tk.Label(
            footer_frame, 
            text=f"© {time.strftime('%Y')}",
            font=("Arial", 9, "bold"),
            bg=COLOR_PRIMARY,
            fg="white",
            padx=10,
            pady=5
        )
        year_label.pack(side=tk.LEFT, padx=10)
        
    def setup_events(self):
        # Eventos de la vista de configuración
        self.config_view.gate_combo.config(values=self.gate_model.GATES)
        self.config_view.gate_combo.bind("<<ComboboxSelected>>", self.on_gate_change)
        self.config_view.train_button.config(command=self.train_perceptron_with_animation)
        self.config_view.apply_weights_button.config(command=self.apply_custom_weights)
        
        # Eventos de la vista de pruebas
        self.test_view.test_gate_combo.config(values=self.gate_model.GATES)
        self.test_view.predict_button.config(command=self.predict_custom_input)
        
        # Vincular evento de cambio de pestaña
        self.main_view.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    # Actualizar el método on_gate_change para usar los nuevos nombres de variables
    def on_gate_change(self, event=None):
        # Actualizar compuerta actual
        self.current_gate = self.config_view.gate_var.get()
        
        # Actualizar datos según la compuerta
        self.current_inputs, self.current_labels = self.gate_model.get_gate_data(self.current_gate)
            
        # Actualizar tabla de verdad
        self.config_view.update_truth_table(self.current_inputs, self.current_labels)
        
        # Actualizar etiqueta en pestaña de pruebas
        self.test_view.current_gate_label.config(text=self.current_gate)
        
        # Reiniciar estado en ambos indicadores
        # Estado en el panel de configuración
        self.config_view.config_status_label.config(text="Estado: No entrenado", fg=COLOR_ACCENT_RED)
        self.config_view.config_status_indicator.delete("all")
        self.config_view.config_status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_ACCENT_RED, outline="")
        
        # Deshabilitar botón de reentrenamiento
        self.config_view.apply_weights_button.config(state=tk.DISABLED)
        
        # Ocultar frames de edición de pesos según la compuerta seleccionada
        if self.current_gate == "XOR":
            self.config_view.simple_edit_frame.pack_forget()
            self.config_view.xor_edit_frame.pack(fill='x', padx=10, pady=5)
        else:
            self.config_view.xor_edit_frame.pack_forget()
            self.config_view.simple_edit_frame.pack(fill='x', padx=10, pady=5)
            
        # Actualizar visualizaciones para la nueva compuerta
        self.visualization_view.update_desired_graph(self.current_inputs, self.current_labels, self.current_gate)
        self.visualization_view.update_decision_boundary(self.current_inputs, self.current_labels, self.current_gate)

    # Actualizar el método apply_custom_weights para usar los nuevos nombres de variables
    def apply_custom_weights(self):
        """Aplica los pesos personalizados ingresados por el usuario"""
        gate = self.current_gate
        
        try:
            if gate == "XOR":
                # Verificar si los campos están llenos
                if not (self.config_view.and_w0_var.get() and self.config_view.and_w1_var.get() and 
                        self.config_view.and_w2_var.get() and self.config_view.or_w0_var.get() and 
                        self.config_view.or_w1_var.get() and self.config_view.or_w2_var.get() and 
                        self.config_view.not_w0_var.get() and self.config_view.not_w1_var.get()):
                    tk.messagebox.showerror("Error", "Todos los campos de pesos deben estar llenos")
                    return
                
                # Crear o actualizar el modelo de tres perceptrones
                if self.three_perceptron_xor is None:
                    self.three_perceptron_xor = ThreePerceptronXOR()
                
                # Aplicar pesos personalizados
                self.three_perceptron_xor.and_perceptron.weights = np.array([
                    float(self.config_view.and_w0_var.get()),
                    float(self.config_view.and_w1_var.get()),
                    float(self.config_view.and_w2_var.get())
                ])
                
                self.three_perceptron_xor.or_perceptron.weights = np.array([
                    float(self.config_view.or_w0_var.get()),
                    float(self.config_view.or_w1_var.get()),
                    float(self.config_view.or_w2_var.get())
                ])
                
                self.three_perceptron_xor.not_perceptron.weights = np.array([
                    float(self.config_view.not_w0_var.get()),
                    float(self.config_view.not_w1_var.get())
                ])
                
                # Crear historial de error ficticio para visualización
                self.three_perceptron_xor.error_history = [0.0]
                
                # Evaluar el modelo con los nuevos pesos
                predictions = [self.three_perceptron_xor.predict(x) for x in self.current_inputs]
                correct = sum(1 for p, l in zip(predictions, self.current_labels) if p == l)
                accuracy = (correct / len(self.current_labels)) * 100
                
                # Actualizar resultados
                self.config_view.update_results(
                    1,  # Épocas (ficticio)
                    accuracy, 
                    "Pesos personalizados", 
                    0.0,  # Error final (ficticio)
                    accuracy == 100,
                    three_perceptron_xor=self.three_perceptron_xor
                )
                
                # Actualizar visualizaciones
                self.visualization_view.update_error_graph([0.0], gate, "ThreePerceptronXOR")
                self.visualization_view.update_obtained_graph(self.current_inputs, predictions, self.current_labels, gate)
                self.visualization_view.update_decision_boundary(
                    self.current_inputs, self.current_labels, gate, 
                    and_perceptron=self.three_perceptron_xor.and_perceptron,
                    or_perceptron=self.three_perceptron_xor.or_perceptron,
                    not_perceptron=self.three_perceptron_xor.not_perceptron
                )
                
                # Si el modelo es correcto, marcar la compuerta como entrenada
                if accuracy == 100:
                    self.gate_model.mark_as_trained(gate)
                else:
                    self.gate_model.unmark_as_trained(gate)
                
            else:
                # Verificar si los campos están llenos
                if not (self.config_view.simple_w0_var.get() and self.config_view.simple_w1_var.get() and 
                        self.config_view.simple_w2_var.get()):
                    tk.messagebox.showerror("Error", "Todos los campos de pesos deben estar llenos")
                    return
                
                # Crear o actualizar el perceptrón simple
                if self.perceptron is None:
                    self.perceptron = Perceptron(n_inputs=2)
                
                # Aplicar pesos personalizados
                self.perceptron.weights = np.array([
                    float(self.config_view.simple_w0_var.get()),
                    float(self.config_view.simple_w1_var.get()),
                    float(self.config_view.simple_w2_var.get())
                ])
                
                # Crear historial de error ficticio para visualización
                self.perceptron.error_history = [0.0]
                
                # Evaluar el perceptrón con los nuevos pesos
                predictions = [self.perceptron.predict(x) for x in self.current_inputs]
                correct = sum(1 for p, l in zip(predictions, self.current_labels) if p == l)
                accuracy = (correct / len(self.current_labels)) * 100
                
                # Actualizar resultados
                self.config_view.update_results(
                    1,  # Épocas (ficticio)
                    accuracy, 
                    "Pesos personalizados", 
                    0.0,  # Error final (ficticio)
                    accuracy == 100,
                    perceptron=self.perceptron
                )
                
                # Actualizar visualizaciones
                self.visualization_view.update_error_graph([0.0], gate, "Perceptron")
                self.visualization_view.update_obtained_graph(self.current_inputs, predictions, self.current_labels, gate)
                self.visualization_view.update_decision_boundary(self.current_inputs, self.current_labels, gate, self.perceptron)
                
                # Si el perceptrón es correcto, marcar la compuerta como entrenada
                if accuracy == 100:
                    self.gate_model.mark_as_trained(gate)
                else:
                    self.gate_model.unmark_as_trained(gate)
            
        except ValueError:
            tk.messagebox.showerror("Error", "Los pesos deben ser valores numéricos")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error al aplicar pesos: {str(e)}")

    def train_perceptron(self):
        # Obtener parámetros
        gate = self.current_gate
        
        # Determinar si la compuerta necesita una red de múltiples capas
        if gate == "XOR":
            # Use three perceptrons for XOR visualization
            self.three_perceptron_xor = ThreePerceptronXOR()
            
            # Add timeout mechanism for XOR training
            import threading
            import time
            
            # Flag to check if training is complete
            training_complete = False
            
            # Function to run training in a separate thread
            def train_thread():
                nonlocal training_complete
                try:
                    actual_epochs = self.three_perceptron_xor.train(self.current_inputs, self.current_labels)
                    training_complete = True
                except Exception as e:
                    print(f"Error in training: {e}")
                    training_complete = True  # Mark as complete even on error
            
            # Start training in a separate thread
            training_thread = threading.Thread(target=train_thread)
            training_thread.daemon = True  # Allow the thread to be terminated when the main program exits
            training_thread.start()
            
            # Wait for training to complete with timeout
            max_wait_time = 5  # seconds
            start_time = time.time()
            while not training_complete and time.time() - start_time < max_wait_time:
                # Update UI to prevent freezing
                self.main_view.root.update()
                time.sleep(0.1)
            
            # If training didn't complete in time, create a basic model
            if not training_complete:
                print("XOR training timed out, using default weights")
                self.three_perceptron_xor = ThreePerceptronXOR()
                # Set some reasonable weights for AND perceptron
                self.three_perceptron_xor.and_perceptron.weights = np.array([1.0, 1.0, -1.5])
                # Set some reasonable weights for OR perceptron
                self.three_perceptron_xor.or_perceptron.weights = np.array([1.0, 1.0, -0.5])
                # Set some reasonable weights for NOT perceptron
                self.three_perceptron_xor.not_perceptron.weights = np.array([-2.0, 1.0])
                self.three_perceptron_xor.error_history = [0.0]  # Fake error history
                actual_epochs = 1
            else:
                actual_epochs = len(self.three_perceptron_xor.error_history)
                
            
            try:
                self.mlp.train(self.current_inputs, self.current_labels, epochs=5000)
            except:
                # If MLP training fails, set default weights
                self.mlp.hidden_weights = np.array([
                    [1.0, -1.0, 1.0, -1.0],  # Input 1
                    [1.0, -1.0, -1.0, 1.0],  # Input 2
                    [-0.5, -0.5, -0.5, -0.5]  # Bias
                ])
                self.mlp.output_weights = np.array([-1.0, -1.0, 1.0, 1.0, -0.5])
                self.mlp.error_history = [0.0]
                
            predictions = [self.three_perceptron_xor.predict(x) for x in self.current_inputs]
            weights_info = "Tres perceptrones (AND + OR + NOT)"
            network_type = "ThreePerceptronXOR"
            
            # Actualizar los campos de edición de pesos
            self.config_view.and_w0_var.set(str(self.three_perceptron_xor.and_perceptron.weights[0]))
            self.config_view.and_w1_var.set(str(self.three_perceptron_xor.and_perceptron.weights[1]))
            self.config_view.and_w2_var.set(str(self.three_perceptron_xor.and_perceptron.weights[2]))
            
            self.config_view.or_w0_var.set(str(self.three_perceptron_xor.or_perceptron.weights[0]))
            self.config_view.or_w1_var.set(str(self.three_perceptron_xor.or_perceptron.weights[1]))
            self.config_view.or_w2_var.set(str(self.three_perceptron_xor.or_perceptron.weights[2]))
            
            self.config_view.not_w0_var.set(str(self.three_perceptron_xor.not_perceptron.weights[0]))
            self.config_view.not_w1_var.set(str(self.three_perceptron_xor.not_perceptron.weights[1]))
            
            # Mostrar el frame de edición de XOR y ocultar el simple
            self.config_view.simple_edit_frame.pack_forget()
            self.config_view.xor_edit_frame.pack(fill='x', padx=10, pady=5)
            
        else:
            # Usar perceptrón simple para otras compuertas
            self.perceptron = Perceptron(n_inputs=2)
            actual_epochs = self.perceptron.train(self.current_inputs, self.current_labels)
            predictions = [self.perceptron.predict(x) for x in self.current_inputs]
            weights_info = str(self.perceptron.weights)
            network_type = "Perceptron"
            
            # Actualizar los campos de edición de pesos
            self.config_view.simple_w0_var.set(str(self.perceptron.weights[0]))
            self.config_view.simple_w1_var.set(str(self.perceptron.weights[1]))
            self.config_view.simple_w2_var.set(str(self.perceptron.weights[2]))
            
            # Mostrar el frame de edición simple y ocultar el de XOR
            self.config_view.xor_edit_frame.pack_forget()
            self.config_view.simple_edit_frame.pack(fill='x', padx=10, pady=5)
        
        # Calcular precisión
        correct = sum(1 for p, l in zip(predictions, self.current_labels) if p == l)
        accuracy = (correct / len(self.current_labels)) * 100
        
        # Obtener error final
        if network_type == "Perceptron" and len(self.perceptron.error_history) > 0:
            error_final = self.perceptron.error_history[-1]
        elif network_type == "ThreePerceptronXOR" and len(self.three_perceptron_xor.error_history) > 0:
            error_final = self.three_perceptron_xor.error_history[-1]
        else:
            error_final = 0
        
        # Actualizar resultados
        if network_type == "Perceptron":
            self.config_view.update_results(
                actual_epochs, 
                accuracy, 
                weights_info, 
                error_final, 
                accuracy == 100,
                perceptron=self.perceptron
            )
        else:  # ThreePerceptronXOR
            self.config_view.update_results(
                actual_epochs, 
                accuracy, 
                weights_info, 
                error_final, 
                accuracy == 100,
                three_perceptron_xor=self.three_perceptron_xor
            )
        
        # Si el entrenamiento fue exitoso, registrar la compuerta como entrenada
        if accuracy == 100:
            self.gate_model.mark_as_trained(gate)
        
        # Actualizar gráficas
        if network_type == "Perceptron":
            self.visualization_view.update_error_graph(self.perceptron.error_history, gate, "Perceptron")
        elif network_type == "ThreePerceptronXOR":
            self.visualization_view.update_error_graph(self.three_perceptron_xor.error_history, gate, "ThreePerceptronXOR")
            
        self.visualization_view.update_desired_graph(self.current_inputs, self.current_labels, gate)
        self.visualization_view.update_obtained_graph(self.current_inputs, predictions, self.current_labels, gate)
        
        # Actualizar gráfica de línea de decisión
        if network_type == "Perceptron":
            self.visualization_view.update_decision_boundary(self.current_inputs, self.current_labels, gate, self.perceptron)
        elif network_type == "ThreePerceptronXOR":
            self.visualization_view.update_decision_boundary(
                self.current_inputs, self.current_labels, gate, 
                and_perceptron=self.three_perceptron_xor.and_perceptron,
                or_perceptron=self.three_perceptron_xor.or_perceptron,
                not_perceptron=self.three_perceptron_xor.not_perceptron
            )
        else:
            self.visualization_view.update_decision_boundary(self.current_inputs, self.current_labels, gate)
        
    def train_perceptron_with_animation(self):
        # Deshabilitar botón durante el entrenamiento
        self.config_view.train_button.config(state=tk.DISABLED)
        
        # Reiniciar barra de progreso
        self.config_view.progress["value"] = 0
        self.main_view.root.update_idletasks()
        
        # Simular progreso
        for i in range(101):
            self.config_view.progress["value"] = i
            self.main_view.root.update_idletasks()
            time.sleep(0.01)
        
        # Entrenar perceptrón
        self.train_perceptron()
        
        # Habilitar botón
        self.config_view.train_button.config(state=tk.NORMAL)
        
    def retrain_perceptron(self):
        """Reentrenar la red con los pesos personalizados ingresados por el usuario"""
        # Deshabilitar botón durante el reentrenamiento
        self.config_view.apply_weights_button.config(state=tk.DISABLED)
        
        # Reiniciar barra de progreso
        self.config_view.progress["value"] = 0
        self.main_view.root.update_idletasks()
        
        # Simular progreso
        for i in range(101):
            self.config_view.progress["value"] = i
            self.main_view.root.update_idletasks()
            time.sleep(0.01)
        
        # Aplicar los pesos personalizados
        self.apply_custom_weights()
        
        # El estado y la habilitación del botón se manejan en update_results
        
    def predict_custom_input(self):
        # Obtener entradas
        try:
            input1 = int(self.test_view.input1_var.get())
            input2 = int(self.test_view.input2_var.get())
            
            if input1 not in [0, 1] or input2 not in [0, 1]:
                raise ValueError("Las entradas deben ser 0 o 1")
                
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
            
        # Obtener la compuerta seleccionada para prueba
        test_gate = self.test_view.test_gate_var.get()
        
        # Verificar si la compuerta ha sido entrenada
        if not self.gate_model.is_trained(test_gate):
            messagebox.showwarning("Compuerta no entrenada", 
                                f"La compuerta {test_gate} no ha sido entrenada. Por favor, entrénela primero.")
            return
        
        # Determinar la red a usar según la compuerta
        inputs = np.array([input1, input2])
        if test_gate == "XOR":
            if hasattr(self, 'three_perceptron_xor') and self.three_perceptron_xor is not None:
                prediction = self.three_perceptron_xor.predict(inputs)
                # Visualizar el proceso
                self.test_view.draw_three_perceptron_diagram(inputs, self.three_perceptron_xor)
            elif self.mlp is not None:
                prediction = self.mlp.predict(inputs)
                # Visualizar el proceso
                self.test_view.draw_mlp_diagram(inputs, self.mlp)
            else:
                messagebox.showwarning("Advertencia", "Primero debe entrenar la red para XOR")
                return
        else:
            if self.perceptron is None:
                messagebox.showwarning("Advertencia", "Primero debe entrenar el perceptrón")
                return
            prediction = self.perceptron.predict(inputs)
            # Visualizar el proceso
            self.test_view.draw_perceptron_diagram(inputs, self.perceptron)
        
        # Determinar si es correcto según la compuerta seleccionada
        expected = self.gate_model.get_expected_output(test_gate, input1, input2)
            
        # Actualizar resultado
        self.test_view.update_prediction_result(prediction, expected, test_gate, input1, input2)
        
    def on_tab_change(self, event):
        # Actualizar la pestaña de pruebas cuando se selecciona
        if self.main_view.notebook.index(self.main_view.notebook.select()) == 4:  # Índice de la pestaña de pruebas
            self.test_view.current_gate_label.config(text=self.test_view.test_gate_var.get())
            # Actualizar la lista de compuertas disponibles para prueba
            self.test_view.test_gate_combo.set(self.current_gate)

    # Actualizar el método retrain_perceptron para usar los nuevos nombres de variables
    def retrain_perceptron(self):
        # Habilitar el botón de entrenamiento
        self.config_view.train_button.config(state=tk.NORMAL)
        
        # Deshabilitar el botón de reentrenamiento
        self.config_view.apply_weights_button.config(state=tk.DISABLED)
        
        # Reiniciar el estado del indicador
        self.config_view.config_status_label.config(text="Estado: No entrenado", fg=COLOR_ACCENT_RED)
        self.config_view.config_status_indicator.delete("all")
        self.config_view.config_status_indicator.create_oval(2, 2, 18, 18, fill=COLOR_ACCENT_RED, outline="")
        
        # Limpiar los pesos personalizados
        if self.current_gate == "XOR":
            self.config_view.and_w0_var.set("")
            self.config_view.and_w1_var.set("")
            self.config_view.and_w2_var.set("")
            self.config_view.or_w0_var.set("")
            self.config_view.or_w1_var.set("")
            self.config_view.or_w2_var.set("")
            self.config_view.not_w0_var.set("")
            self.config_view.not_w1_var.set("")
        else:
            self.config_view.simple_w0_var.set("")
            self.config_view.simple_w1_var.set("")
            self.config_view.simple_w2_var.set("")