import tkinter as tk
from tkinter import ttk
import numpy as np
from utils.ui_components import (
    COLOR_BG, COLOR_LIGHT_BG, COLOR_PRIMARY, COLOR_SECONDARY, 
    COLOR_TEXT, COLOR_ACCENT_RED, ToolTip, AnimatedButton
)

class TestView:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.setup_test_tab()
        
    def setup_test_tab(self):
        # Frame principal para pruebas
        test_main = tk.Frame(self.parent, bg=COLOR_BG)
        test_main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Panel de pruebas
        test_panel = tk.Frame(test_main, bg=COLOR_LIGHT_BG, bd=1, relief=tk.GROOVE)
        test_panel.pack(fill='both', expand=True)
        
        test_title = tk.Label(test_panel, text="Prueba de Compuerta Lógica", 
                            font=("Arial", 16, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        test_title.pack(pady=(20, 10))
        
        test_subtitle = tk.Label(test_panel, 
                               text="Ingrese valores de entrada (0 o 1) y obtenga la predicción del perceptrón", 
                               font=("Arial", 12), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        test_subtitle.pack(pady=(0, 20))
        
        # Lista desplegable de todas las compuertas lógicas para probar
        gate_test_frame = tk.Frame(test_panel, bg=COLOR_LIGHT_BG)
        gate_test_frame.pack(pady=10)
        
        gate_test_label = tk.Label(gate_test_frame, text="Seleccionar compuerta a probar:", 
                                font=("Arial", 12, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        gate_test_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_gate_var = tk.StringVar(value="AND")
        self.test_gate_combo = ttk.Combobox(gate_test_frame, textvariable=self.test_gate_var, 
                                         width=15, font=("Arial", 12), state="readonly")
        self.test_gate_combo.pack(side=tk.LEFT)
        
        # Separador
        separator = ttk.Separator(test_panel, orient='horizontal')
        separator.pack(fill='x', padx=50, pady=10)
        
        # Contenedor para entradas
        input_container = tk.Frame(test_panel, bg=COLOR_LIGHT_BG)
        input_container.pack(pady=20)
        
        # Entrada 1
        input1_frame = tk.Frame(input_container, bg=COLOR_LIGHT_BG)
        input1_frame.pack(side=tk.LEFT, padx=20)
        
        input1_label = tk.Label(input1_frame, text="Entrada 1", 
                              font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        input1_label.pack(pady=5)
        
        self.input1_var = tk.StringVar(value="0")
        input1_combo = ttk.Combobox(input1_frame, textvariable=self.input1_var, 
                                   values=["0", "1"], width=5, font=("Arial", 14), state="readonly")
        input1_combo.pack()
        
        # Entrada 2
        input2_frame = tk.Frame(input_container, bg=COLOR_LIGHT_BG)
        input2_frame.pack(side=tk.LEFT, padx=20)
        
        input2_label = tk.Label(input2_frame, text="Entrada 2", 
                              font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        input2_label.pack(pady=5)
        
        self.input2_var = tk.StringVar(value="0")
        input2_combo = ttk.Combobox(input2_frame, textvariable=self.input2_var, 
                                   values=["0", "1"], width=5, font=("Arial", 14), state="readonly")
        input2_combo.pack()
        
        # Operador
        operator_frame = tk.Frame(input_container, bg=COLOR_LIGHT_BG)
        operator_frame.pack(side=tk.LEFT, padx=20)
        
        self.current_gate_label = tk.Label(operator_frame, text="AND", 
                                         font=("Arial", 18, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        self.current_gate_label.pack(pady=5)
        
        gate_desc = tk.Label(operator_frame, text="Compuerta", 
                           font=("Arial", 10), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        gate_desc.pack()
        
        # Botón de predicción
        predict_frame = tk.Frame(test_panel, bg=COLOR_LIGHT_BG)
        predict_frame.pack(pady=20)
        
        self.predict_button = AnimatedButton(
            predict_frame, text="Predecir Salida", 
            font=("Arial", 12, "bold"), 
            bg=COLOR_PRIMARY, fg="white",
            hover_bg=COLOR_SECONDARY, hover_fg=COLOR_PRIMARY,
            activebackground=COLOR_SECONDARY, activeforeground=COLOR_PRIMARY,
            padx=20, pady=10, relief=tk.RAISED, bd=2
        )
        self.predict_button.pack()
        
        # Separador
        separator2 = ttk.Separator(test_panel, orient='horizontal')
        separator2.pack(fill='x', padx=50, pady=20)
        
        # Resultado de la predicción
        result_container = tk.Frame(test_panel, bg=COLOR_LIGHT_BG)
        result_container.pack(pady=20)
        
        result_label = tk.Label(result_container, text="Resultado:", 
                              font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        result_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.prediction_result = tk.Label(result_container, text="--", 
                                        font=("Arial", 24, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_ACCENT_RED)
        self.prediction_result.pack(side=tk.LEFT)
        
        # Explicación
        self.explanation_var = tk.StringVar(value="Realice una predicción para ver el resultado")
        explanation = tk.Label(test_panel, textvariable=self.explanation_var, 
                             font=("Arial", 12, "italic"), bg=COLOR_LIGHT_BG, fg=COLOR_TEXT)
        explanation.pack(pady=10)
        
        # Visualización del proceso
        visual_frame = tk.Frame(test_panel, bg=COLOR_LIGHT_BG)
        visual_frame.pack(fill='x', padx=50, pady=20)
        
        visual_title = tk.Label(visual_frame, text="Visualización del Proceso", 
                              font=("Arial", 14, "bold"), bg=COLOR_LIGHT_BG, fg=COLOR_PRIMARY)
        visual_title.pack(pady=(0, 10))
        
        self.process_canvas = tk.Canvas(visual_frame, width=600, height=150, 
                                      bg="white", highlightthickness=1, highlightbackground=COLOR_PRIMARY)
        self.process_canvas.pack()
        
    def update_prediction_result(self, prediction, expected, gate_name, input1, input2):
        # Actualizar resultado
        self.prediction_result.config(text=str(prediction))
        
        # Actualizar color según si es correcto
        if prediction == expected:
            self.prediction_result.config(fg=COLOR_PRIMARY)
            self.explanation_var.set(f"¡Correcto! La compuerta {gate_name} con entradas {input1} y {input2} debe dar {expected}")
        else:
            self.prediction_result.config(fg=COLOR_ACCENT_RED)
            self.explanation_var.set(f"¡Incorrecto! La compuerta {gate_name} con entradas {input1} y {input2} debe dar {expected}")
            
    def draw_perceptron_diagram(self, inputs, perceptron):
        # Limpiar canvas
        self.process_canvas.delete("all")
        
        # Entradas
        input_x = 50
        input1_y = 50
        input2_y = 100
        
        # Neurona
        neuron_x = 300
        neuron_y = 75
        neuron_radius = 30
        
        # Salida
        output_x = 500
        output_y = 75
        
        # Dibujar entradas
        self.process_canvas.create_text(input_x - 30, input1_y, text="X1", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input1_y, text=str(inputs[0]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input1_y - 15, input_x + 15, input1_y + 15, 
                                      fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        self.process_canvas.create_text(input_x - 30, input2_y, text="X2", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input2_y, text=str(inputs[1]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input2_y - 15, input_x + 15, input2_y + 15, 
                                      fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # Dibujar neurona
        self.process_canvas.create_oval(neuron_x - neuron_radius, neuron_y - neuron_radius, 
                                      neuron_x + neuron_radius, neuron_y + neuron_radius, 
                                      fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(neuron_x, neuron_y, text="Σ", font=("Arial", 16, "bold"))
        
        # Dibujar conexiones con pesos
        # Conexión 1
        self.process_canvas.create_line(input_x + 15, input1_y, neuron_x - neuron_radius, neuron_y - 10, 
                                      fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        w1_text = f"w1 = {perceptron.weights[0]}"
        self.process_canvas.create_text((input_x + 15 + neuron_x - neuron_radius) / 2, 
                                      (input1_y + neuron_y - 10) / 2 - 10, 
                                      text=w1_text, font=("Arial", 10))
        
        # Conexión 2
        self.process_canvas.create_line(input_x + 15, input2_y, neuron_x - neuron_radius, neuron_y + 10, 
                                      fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        w2_text = f"w2 = {perceptron.weights[1]}"
        self.process_canvas.create_text((input_x + 15 + neuron_x - neuron_radius) / 2, 
                                      (input2_y + neuron_y + 10) / 2 + 10, 
                                      text=w2_text, font=("Arial", 10))
        
        # Bias
        bias_x = neuron_x
        bias_y = neuron_y - neuron_radius - 20
        self.process_canvas.create_text(bias_x, bias_y, text="Bias = 1", font=("Arial", 10))
        self.process_canvas.create_line(bias_x, bias_y + 10, neuron_x, neuron_y - neuron_radius, 
                                      fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        w3_text = f"w3 = {perceptron.weights[2]}"
        self.process_canvas.create_text(bias_x + 15, bias_y + 15, text=w3_text, font=("Arial", 10))
        
        # Dibujar salida
        prediction = perceptron.predict(inputs)
        self.process_canvas.create_line(neuron_x + neuron_radius, neuron_y, output_x - 15, output_y, 
                                      fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        self.process_canvas.create_text((neuron_x + neuron_radius + output_x - 15) / 2, 
                                      output_y - 15, text="Activación", font=("Arial", 10))
        
        output_color = COLOR_PRIMARY if prediction == 1 else COLOR_ACCENT_RED
        self.process_canvas.create_oval(output_x - 15, output_y - 15, output_x + 15, output_y + 15, 
                                      fill=output_color, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(output_x, output_y, text=str(prediction), 
                                      font=("Arial", 12, "bold"), fill="white")
        self.process_canvas.create_text(output_x + 40, output_y, text="Salida", 
                                      font=("Arial", 12, "bold"))
        
        # Mostrar cálculo
        calc_text = f"Σ = ({inputs[0]} × {perceptron.weights[0]}) + ({inputs[1]} × {perceptron.weights[1]}) + (1 × {perceptron.weights[2]})"
        sum_value = np.dot(np.append(inputs, 1), perceptron.weights)
        self.process_canvas.create_text(neuron_x, neuron_y + neuron_radius + 20, 
                                      text=calc_text, font=("Arial", 10))
        self.process_canvas.create_text(neuron_x, neuron_y + neuron_radius + 40, 
                                      text=f"Σ = {sum_value}", font=("Arial", 10, "bold"))
        self.process_canvas.create_text(neuron_x, neuron_y + neuron_radius + 60, 
                                      text=f"Activación = 1 si Σ ≥ 0, sino 0", font=("Arial", 10))

        # Mostrar pesos del perceptrón de forma más visible
        weights_text = f"Pesos: w0={perceptron.weights[0]}, w1={perceptron.weights[1]}, w2={perceptron.weights[2]}"
        self.process_canvas.create_text(neuron_x, neuron_y - neuron_radius - 40, 
                                  text=weights_text, font=("Arial", 10, "bold"),
                                  fill=COLOR_PRIMARY)
    
    def draw_mlp_diagram(self, inputs, mlp):
        # Limpiar canvas
        self.process_canvas.delete("all")
        
        # Simplificado para mostrar la arquitectura de red multicapa
        canvas_width = 600
        canvas_height = 150
        
        # Coordenadas
        input_x = 50
        input1_y = 40
        input2_y = 110
        bias_y = 75
        
        hidden_x = 200
        hidden_y_list = [30, 60, 90, 120]
        
        output_x = 350
        output_y = 75
        
        result_x = 500
        result_y = 75
        
        # Dibujar capa de entrada
        self.process_canvas.create_text(input_x - 30, 20, text="Entradas", font=("Arial", 10, "bold"))
        
        # X1
        self.process_canvas.create_text(input_x - 30, input1_y, text="X1", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input1_y, text=str(inputs[0]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input1_y - 15, input_x + 15, input1_y + 15, 
                                      fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # X2
        self.process_canvas.create_text(input_x - 30, input2_y, text="X2", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input2_y, text=str(inputs[1]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input2_y - 15, input_x + 15, input2_y + 15, 
                                      fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # Bias
        self.process_canvas.create_text(input_x, bias_y, text="Bias", font=("Arial", 10))
        self.process_canvas.create_oval(input_x - 15, bias_y - 15, input_x + 15, bias_y + 15, 
                                      fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2, dash=(2, 2))
        
        # Dibujar capa oculta
        self.process_canvas.create_text(hidden_x - 30, 10, text="Capa Oculta", font=("Arial", 10, "bold"))
        
        for i, y in enumerate(hidden_y_list):
            self.process_canvas.create_oval(hidden_x - 15, y - 15, hidden_x + 15, y + 15, 
                                          fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
            self.process_canvas.create_text(hidden_x, y, text=f"H{i+1}", font=("Arial", 10, "bold"))
            
            # Conectar con capa de entrada
            self.process_canvas.create_line(input_x + 15, input1_y, hidden_x - 15, y, 
                                          fill=COLOR_PRIMARY, width=1)
            self.process_canvas.create_line(input_x + 15, input2_y, hidden_x - 15, y, 
                                          fill=COLOR_PRIMARY, width=1)
            self.process_canvas.create_line(input_x + 15, bias_y, hidden_x - 15, y, 
                                          fill=COLOR_PRIMARY, width=1, dash=(2, 2))
        
        # Dibujar neurona de salida
        self.process_canvas.create_text(output_x - 30, 20, text="Salida", font=("Arial", 10, "bold"))
        self.process_canvas.create_oval(output_x - 15, output_y - 15, output_x + 15, output_y + 15, 
                                     fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(output_x, output_y, text="O", font=("Arial", 12, "bold"))
        
        # Conectar capa oculta con salida
        for y in hidden_y_list:
            self.process_canvas.create_line(hidden_x + 15, y, output_x - 15, output_y, 
                                         fill=COLOR_PRIMARY, width=1)
        
        # Bias para capa de salida
        bias2_x = hidden_x
        bias2_y = 140
        self.process_canvas.create_text(bias2_x, bias2_y, text="Bias", font=("Arial", 10))
        self.process_canvas.create_oval(bias2_x - 15, bias2_y - 15, bias2_x + 15, bias2_y + 15, 
                                     fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2, dash=(2, 2))
        self.process_canvas.create_line(bias2_x + 15, bias2_y, output_x - 15, output_y, 
                                     fill=COLOR_PRIMARY, width=1, dash=(2, 2))
        
        # Resultado final
        prediction = mlp.predict(inputs)  # Use the actual prediction from the MLP model
        self.process_canvas.create_line(output_x + 15, output_y, result_x - 15, result_y, 
                                     fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        
        output_color = COLOR_PRIMARY if prediction == 1 else COLOR_ACCENT_RED
        self.process_canvas.create_oval(result_x - 15, result_y - 15, result_x + 15, result_y + 15, 
                                     fill=output_color, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(result_x, result_y, text=str(prediction), 
                                     font=("Arial", 12, "bold"), fill="white")
        self.process_canvas.create_text(result_x + 40, result_y, text="Salida", 
                                     font=("Arial", 12, "bold"))
        
        # Nota explicativa
        self.process_canvas.create_text(canvas_width/2, canvas_height - 10, 
                                     text="Red neuronal multicapa para resolver XOR (diagrama simplificado)", 
                                     font=("Arial", 10, "italic"))

    # Add a new method to draw the three perceptron diagram for XOR
    def draw_three_perceptron_diagram(self, inputs, three_perceptron_xor):
        # Limpiar canvas
        self.process_canvas.delete("all")
        
        # Simplificado para mostrar la arquitectura de tres perceptrones
        canvas_width = 600
        canvas_height = 150
        
        # Coordenadas
        input_x = 50
        input1_y = 40
        input2_y = 110
        
        # Perceptrones
        and_x = 150
        and_y = 50
        
        or_x = 150
        or_y = 100
        
        not_x = 250
        not_y = 50
        
        # Salida AND final
        final_and_x = 350
        final_and_y = 75
        
        # Resultado final
        result_x = 500
        result_y = 75
        
        # Dibujar capa de entrada
        self.process_canvas.create_text(input_x - 30, 20, text="Entradas", font=("Arial", 10, "bold"))
        
        # X1
        self.process_canvas.create_text(input_x - 30, input1_y, text="X1", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input1_y, text=str(inputs[0]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input1_y - 15, input_x + 15, input1_y + 15, 
                                    fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # X2
        self.process_canvas.create_text(input_x - 30, input2_y, text="X2", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input2_y, text=str(inputs[1]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input2_y - 15, input_x + 15, input2_y + 15, 
                                    fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # Dibujar perceptrón AND
        self.process_canvas.create_oval(and_x - 15, and_y - 15, and_x + 15, and_y + 15, 
                                   fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(and_x, and_y, text="AND", font=("Arial", 10, "bold"))

        # Mostrar pesos del perceptrón AND
        and_weights = three_perceptron_xor.and_perceptron.weights
        and_weights_text = f"w0={and_weights[0]}, w1={and_weights[1]}, w2={and_weights[2]}"
        self.process_canvas.create_text(and_x, and_y + 20, text=and_weights_text, font=("Arial", 8))
        
        # Dibujar perceptrón OR
        self.process_canvas.create_oval(or_x - 15, or_y - 15, or_x + 15, or_y + 15, 
                                    fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(or_x, or_y, text="OR", font=("Arial", 10, "bold"))

        # Mostrar pesos del perceptrón OR
        or_weights = three_perceptron_xor.or_perceptron.weights
        or_weights_text = f"w0={or_weights[0]}, w1={or_weights[1]}, w2={or_weights[2]}"
        self.process_canvas.create_text(or_x, or_y + 20, text=or_weights_text, font=("Arial", 8))
        
        # Conectar entradas con perceptrones AND y OR
        self.process_canvas.create_line(input_x + 15, input1_y, and_x - 15, and_y, 
                                   fill=COLOR_PRIMARY, width=1)
        self.process_canvas.create_line(input_x + 15, input2_y, and_x - 15, and_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        self.process_canvas.create_line(input_x + 15, input1_y, or_x - 15, or_y, 
                                   fill=COLOR_PRIMARY, width=1)
        self.process_canvas.create_line(input_x + 15, input2_y, or_x - 15, or_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        # Calcular salidas de AND y OR
        and_output = three_perceptron_xor.and_perceptron.predict(inputs)
        or_output = three_perceptron_xor.or_perceptron.predict(inputs)
        
        # Mostrar salidas intermedias
        self.process_canvas.create_text(and_x + 30, and_y, text=f"→ {and_output}", font=("Arial", 10))
        self.process_canvas.create_text(or_x + 30, or_y, text=f"→ {or_output}", font=("Arial", 10))
        
        # Dibujar perceptrón NOT
        self.process_canvas.create_oval(not_x - 15, not_y - 15, not_x + 15, not_y + 15, 
                                    fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(not_x, not_y, text="NOT", font=("Arial", 10, "bold"))

        # Mostrar pesos del perceptrón NOT
        not_weights = three_perceptron_xor.not_perceptron.weights
        not_weights_text = f"w0={not_weights[0]}, w1={not_weights[1]}"
        self.process_canvas.create_text(not_x, not_y + 20, text=not_weights_text, font=("Arial", 8))
        
        # Conectar AND con NOT
        self.process_canvas.create_line(and_x + 15, and_y, not_x - 15, not_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        # Calcular salida de NOT
        not_and_output = three_perceptron_xor.not_perceptron.predict(np.array([and_output]))
        
        # Mostrar salida intermedia de NOT
        self.process_canvas.create_text(not_x + 30, not_y, text=f"→ {not_and_output}", font=("Arial", 10))
        
        # Dibujar neurona AND final para combinar salidas
        self.process_canvas.create_oval(final_and_x - 15, final_and_y - 15, final_and_x + 15, final_and_y + 15, 
                                   fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(final_and_x, final_and_y, text="AND", font=("Arial", 10, "bold"))
        
        # Conectar OR y NOT con AND final
        self.process_canvas.create_line(or_x + 15, or_y, final_and_x - 15, final_and_y, 
                                   fill=COLOR_PRIMARY, width=1)
        self.process_canvas.create_line(not_x + 15, not_y, final_and_x - 15, final_and_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        # Resultado final
        prediction = three_perceptron_xor.predict(inputs)
        self.process_canvas.create_line(final_and_x + 15, final_and_y, result_x - 15, result_y, 
                                   fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        
        output_color = COLOR_PRIMARY if prediction == 1 else COLOR_ACCENT_RED
        self.process_canvas.create_oval(result_x - 15, result_y - 15, result_x + 15, result_y + 15, 
                                   fill=output_color, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(result_x, result_y, text=str(prediction), 
                                   font=("Arial", 12, "bold"), fill="white")
        self.process_canvas.create_text(result_x + 40, result_y, text="Salida", 
                                   font=("Arial", 12, "bold"))
        
        # Nota explicativa
        self.process_canvas.create_text(canvas_width/2, canvas_height - 10, 
                                   text="XOR implementado con tres perceptrones: AND, OR y NOT(AND)", 
                                   font=("Arial", 10, "italic"))

    # Add a new method to draw the two perceptron diagram for XOR
    def draw_two_perceptron_diagram(self, inputs, two_perceptron_xor):
        # Limpiar canvas
        self.process_canvas.delete("all")
        
        # Simplificado para mostrar la arquitectura de dos perceptrones
        canvas_width = 600
        canvas_height = 150
        
        # Coordenadas
        input_x = 50
        input1_y = 40
        input2_y = 110
        bias_y = 75
        
        # Perceptrones
        or_x = 200
        or_y = 50
        
        nand_x = 200
        nand_y = 100
        
        # Salida AND
        and_x = 350
        and_y = 75
        
        # Resultado final
        result_x = 500
        result_y = 75
        
        # Dibujar capa de entrada
        self.process_canvas.create_text(input_x - 30, 20, text="Entradas", font=("Arial", 10, "bold"))
        
        # X1
        self.process_canvas.create_text(input_x - 30, input1_y, text="X1", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input1_y, text=str(inputs[0]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input1_y - 15, input_x + 15, input1_y + 15, 
                                    fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # X2
        self.process_canvas.create_text(input_x - 30, input2_y, text="X2", font=("Arial", 12, "bold"))
        self.process_canvas.create_text(input_x, input2_y, text=str(inputs[1]), font=("Arial", 12))
        self.process_canvas.create_oval(input_x - 15, input2_y - 15, input_x + 15, input2_y + 15, 
                                    fill=COLOR_LIGHT_BG, outline=COLOR_PRIMARY, width=2)
        
        # Dibujar perceptrón OR
        self.process_canvas.create_oval(or_x - 15, or_y - 15, or_x + 15, or_y + 15, 
                                   fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(or_x, or_y, text="OR", font=("Arial", 10, "bold"))
        
        # Dibujar perceptrón NAND
        self.process_canvas.create_oval(nand_x - 15, nand_y - 15, nand_x + 15, nand_y + 15, 
                                    fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(nand_x, nand_y, text="NAND", font=("Arial", 10, "bold"))
        
        # Conectar entradas con perceptrones
        self.process_canvas.create_line(input_x + 15, input1_y, or_x - 15, or_y, 
                                   fill=COLOR_PRIMARY, width=1)
        self.process_canvas.create_line(input_x + 15, input2_y, or_x - 15, or_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        self.process_canvas.create_line(input_x + 15, input1_y, nand_x - 15, nand_y, 
                                   fill=COLOR_PRIMARY, width=1)
        self.process_canvas.create_line(input_x + 15, input2_y, nand_x - 15, nand_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        # Dibujar neurona AND para combinar salidas
        self.process_canvas.create_oval(and_x - 15, and_y - 15, and_x + 15, and_y + 15, 
                                   fill=COLOR_SECONDARY, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(and_x, and_y, text="AND", font=("Arial", 10, "bold"))
        
        # Conectar perceptrones con AND
        self.process_canvas.create_line(or_x + 15, or_y, and_x - 15, and_y, 
                                   fill=COLOR_PRIMARY, width=1)
        self.process_canvas.create_line(nand_x + 15, nand_y, and_x - 15, and_y, 
                                   fill=COLOR_PRIMARY, width=1)
        
        # Calcular salidas de cada perceptrón
        or_output = two_perceptron_xor.or_perceptron.predict(inputs)
        nand_output = two_perceptron_xor.nand_perceptron.predict(inputs)
        
        # Mostrar salidas intermedias
        self.process_canvas.create_text(or_x + 30, or_y, text=f"→ {or_output}", font=("Arial", 10))
        self.process_canvas.create_text(nand_x + 30, nand_y, text=f"→ {nand_output}", font=("Arial", 10))
        
        # Resultado final
        prediction = two_perceptron_xor.predict(inputs)
        self.process_canvas.create_line(and_x + 15, and_y, result_x - 15, result_y, 
                                   fill=COLOR_PRIMARY, width=2, arrow=tk.LAST)
        
        output_color = COLOR_PRIMARY if prediction == 1 else COLOR_ACCENT_RED
        self.process_canvas.create_oval(result_x - 15, result_y - 15, result_x + 15, result_y + 15, 
                                   fill=output_color, outline=COLOR_PRIMARY, width=2)
        self.process_canvas.create_text(result_x, result_y, text=str(prediction), 
                                   font=("Arial", 12, "bold"), fill="white")
        self.process_canvas.create_text(result_x + 40, result_y, text="Salida", 
                                   font=("Arial", 12, "bold"))
        
        # Nota explicativa
        self.process_canvas.create_text(canvas_width/2, canvas_height - 10, 
                                   text="XOR implementado con dos perceptrones: OR y NAND combinados con AND", 
                                   font=("Arial", 10, "italic"))