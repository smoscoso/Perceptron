import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs + 1) * 2 - 1  # +1 para el sesgo
        self.learning_rate = 0.01  # Tasa de aprendizaje fija internamente
        self.error_history = []
        
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        # Añadir sesgo
        inputs = np.append(inputs, 1)
        summation = np.dot(inputs, self.weights)
        return self.activation(summation)
    
    def train(self, training_inputs, labels, epochs=1000000):
        self.error_history = []
        
        for epoch in range(epochs):
            total_error = 0
            
            for inputs, label in zip(training_inputs, labels):
                # Añadir sesgo
                inputs_with_bias = np.append(inputs, 1)
                
                # Predicción
                prediction = self.predict(inputs)
                
                # Calcular error
                error = label - prediction
                total_error += abs(error)
                
                # Actualizar pesos
                self.weights += self.learning_rate * error * inputs_with_bias
            
            self.error_history.append(total_error)
            
            # Detener si el error es 0
            if total_error == 0:
                break
        
        return len(self.error_history)  # Retorna el número de épocas reales

class MultiLayerPerceptron:
    def __init__(self, n_inputs, n_hidden=4):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        
        # Pesos capa oculta (incluye bias)
        self.hidden_weights = np.random.rand(n_inputs + 1, n_hidden) * 2 - 1
        
        # Pesos capa salida (incluye bias)
        self.output_weights = np.random.rand(n_hidden + 1) * 2 - 1
        
        self.learning_rate = 0.1  # Fijo internamente
        self.error_history = []
        
    def sigmoid(self, x):
        # Add clipping to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        # Forward pass
        # Añadir bias a la entrada
        inputs_with_bias = np.append(inputs, 1)
        
        # Calcular salidas de capa oculta
        hidden_outputs_raw = np.dot(inputs_with_bias, self.hidden_weights)
        hidden_outputs = self.sigmoid(hidden_outputs_raw)
        
        # Añadir bias a las salidas de capa oculta
        hidden_outputs_with_bias = np.append(hidden_outputs, 1)
        
        # Calcular salida final
        output = np.dot(hidden_outputs_with_bias, self.output_weights)
        output = self.sigmoid(output)
        
        # Binarizar la salida para compuertas lógicas
        return 1 if output >= 0.5 else 0
    
    def train(self, training_inputs, labels, epochs=10000):  # Reduced max epochs
        self.error_history = []
        
        for epoch in range(epochs):
            total_error = 0
            
            # Para cada patrón de entrenamiento
            for inputs, target in zip(training_inputs, labels):
                # Forward pass
                # Añadir bias a la entrada
                inputs_with_bias = np.append(inputs, 1)
                
                # Calcular salidas de capa oculta
                hidden_outputs_raw = np.dot(inputs_with_bias, self.hidden_weights)
                hidden_outputs = self.sigmoid(hidden_outputs_raw)
                
                # Añadir bias a las salidas de capa oculta
                hidden_outputs_with_bias = np.append(hidden_outputs, 1)
                
                # Calcular salida final
                output_raw = np.dot(hidden_outputs_with_bias, self.output_weights)
                output = self.sigmoid(output_raw)
                
                # Calcular error
                error = target - output
                total_error += abs(error)
                
                # Backpropagation
                # Error en la capa de salida
                output_delta = error * self.sigmoid_derivative(output)
                
                # Error en la capa oculta
                hidden_error = output_delta * self.output_weights[:-1]  # Excluir bias
                hidden_delta = hidden_error * self.sigmoid_derivative(hidden_outputs)
                
                # Actualizar pesos de la capa de salida
                for i in range(len(self.output_weights)):
                    if i < len(hidden_outputs):
                        self.output_weights[i] += self.learning_rate * output_delta * hidden_outputs[i]
                    else:  # Bias
                        self.output_weights[i] += self.learning_rate * output_delta
                
                # Actualizar pesos de la capa oculta
                for i in range(len(inputs_with_bias)):
                    for j in range(len(hidden_delta)):
                        self.hidden_weights[i, j] += self.learning_rate * hidden_delta[j] * inputs_with_bias[i]
            
            # Record error every 10 epochs to reduce memory usage
            if epoch % 10 == 0:
                self.error_history.append(total_error)
            
            # Detener si el error es muy pequeño
            if total_error < 0.01:
                # Add final error if not already added
                if epoch % 10 != 0:
                    self.error_history.append(total_error)
                break
                
            # Check if we're making progress every 1000 epochs
            if epoch > 0 and epoch % 1000 == 0:
                if len(self.error_history) >= 2:
                    # If error isn't decreasing significantly, increase learning rate
                    if self.error_history[-1] > self.error_history[-2] * 0.99:
                        self.learning_rate *= 1.1
                    # If error is decreasing well, slightly decrease learning rate for stability
                    else:
                        self.learning_rate *= 0.95
        
        # Ensure we have at least one error value
        if not self.error_history:
            self.error_history.append(total_error)
            
        return len(self.error_history) * 10  # Approximate epochs

# Add a new class for handling XOR with three perceptrons
class ThreePerceptronXOR:
    def __init__(self):
        # Create three perceptrons - one for AND, one for OR, and one for NOT
        self.and_perceptron = Perceptron(n_inputs=2)
        self.or_perceptron = Perceptron(n_inputs=2)
        self.not_perceptron = Perceptron(n_inputs=1)  # NOT only needs one input
        self.error_history = []
        
    def train(self, training_inputs, labels, epochs=10000):
        # Training data for AND gate
        and_inputs = training_inputs
        and_labels = np.array([0, 0, 0, 1])  # AND gate outputs
        
        # Training data for OR gate
        or_inputs = training_inputs
        or_labels = np.array([0, 1, 1, 1])  # OR gate outputs
        
        # Training data for NOT gate (using the output of AND as input)
        not_inputs = np.array([[0], [0], [0], [1]])  # Output of AND gate
        not_labels = np.array([1, 1, 1, 0])  # NOT gate outputs
        
        # Train all three perceptrons
        self.and_perceptron.train(and_inputs, and_labels, epochs)
        self.or_perceptron.train(or_inputs, or_labels, epochs)
        self.not_perceptron.train(not_inputs, not_labels, epochs)
        
        # Combine error histories for visualization
        max_len = max(len(self.and_perceptron.error_history), 
                      len(self.or_perceptron.error_history),
                      len(self.not_perceptron.error_history))
        combined_errors = []
        
        for i in range(max_len):
            and_error = self.and_perceptron.error_history[i] if i < len(self.and_perceptron.error_history) else 0
            or_error = self.or_perceptron.error_history[i] if i < len(self.or_perceptron.error_history) else 0
            not_error = self.not_perceptron.error_history[i] if i < len(self.not_perceptron.error_history) else 0
            combined_errors.append(and_error + or_error + not_error)
        
        self.error_history = combined_errors
        return max_len
    
    def predict(self, inputs):
        # Get outputs from the perceptrons
        or_output = self.or_perceptron.predict(inputs)
        and_output = self.and_perceptron.predict(inputs)
        not_and_output = self.not_perceptron.predict(np.array([and_output]))
        
        # XOR is OR AND NOT(AND)
        # This is equivalent to: (A OR B) AND NOT(A AND B)
        return 1 if or_output == 1 and not_and_output == 1 else 0