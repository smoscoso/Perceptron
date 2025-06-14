import numpy as np

class TestController:
    def __init__(self, gate_model, training_controller):
        self.gate_model = gate_model
        self.training_controller = training_controller
        
    def predict(self, gate_name, input1, input2):
        # Verificar si la compuerta ha sido entrenada
        if not self.gate_model.is_trained(gate_name):
            return {
                'success': False,
                'message': f"La compuerta {gate_name} no ha sido entrenada. Por favor, entrénela primero."
            }
        
        # Obtener la red entrenada
        network = self.training_controller.get_network(gate_name)
        
        if network is None:
            return {
                'success': False,
                'message': f"No se ha encontrado una red entrenada para la compuerta {gate_name}."
            }
        
        # Realizar predicción
        inputs = np.array([input1, input2])
        prediction = network.predict(inputs)
        
        # Obtener salida esperada
        expected = self.gate_model.get_expected_output(gate_name, input1, input2)
        
        # Determinar si es correcto
        is_correct = prediction == expected
        
        return {
            'success': True,
            'prediction': prediction,
            'expected': expected,
            'is_correct': is_correct,
            'network': network,
            'inputs': inputs
        }