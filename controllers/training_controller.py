import numpy as np
from models.perceptron_model import Perceptron, MultiLayerPerceptron

class TrainingController:
    def __init__(self, gate_model):
        self.gate_model = gate_model
        self.perceptron = None
        self.mlp = None
        
    def train_network(self, gate_name):
        # Obtener datos de entrenamiento
        inputs, labels = self.gate_model.get_gate_data(gate_name)
        
        # Determinar si la compuerta necesita una red de múltiples capas
        if gate_name == "XOR":
            # Usar MLP para XOR
            self.mlp = MultiLayerPerceptron(n_inputs=2, n_hidden=4)
            actual_epochs = self.mlp.train(inputs, labels)
            predictions = [self.mlp.predict(x) for x in inputs]
            weights_info = "Red neuronal multicapa"
            network_type = "MLP"
            error_history = self.mlp.error_history
        else:
            # Usar perceptrón simple para otras compuertas
            self.perceptron = Perceptron(n_inputs=2)
            actual_epochs = self.perceptron.train(inputs, labels)
            predictions = [self.perceptron.predict(x) for x in inputs]
            weights_info = str(self.perceptron.weights)
            network_type = "Perceptron"
            error_history = self.perceptron.error_history
        
        # Calcular precisión
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = (correct / len(labels)) * 100
        
        # Obtener error final
        error_final = error_history[-1] if error_history else 0
        
        # Si el entrenamiento fue exitoso, registrar la compuerta como entrenada
        if accuracy == 100:
            self.gate_model.mark_as_trained(gate_name)
            
        # Devolver resultados
        return {
            'epochs': actual_epochs,
            'accuracy': accuracy,
            'weights_info': weights_info,
            'error_final': error_final,
            'predictions': predictions,
            'network_type': network_type,
            'error_history': error_history
        }
        
    def get_network(self, gate_name):
        if gate_name == "XOR":
            return self.mlp
        else:
            return self.perceptron