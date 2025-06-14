import numpy as np

class GateModel:
    """
    Modelo para representar el estado de las compuertas lógicas.
    """

    def __init__(self):
        # Datos de entrenamiento para las compuertas lógicas
        self.AND_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.AND_labels = np.array([0, 0, 0, 1])

        self.OR_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.OR_labels = np.array([0, 1, 1, 1])

        self.NAND_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.NAND_labels = np.array([1, 1, 1, 0])

        self.NOR_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.NOR_labels = np.array([1, 0, 0, 0])

        self.XOR_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.XOR_labels = np.array([0, 1, 1, 0])
        
        # Lista completa de compuertas
        self.GATES = ["AND", "OR", "NAND", "NOR", "XOR"]
        
        # Registro de compuertas entrenadas
        self.trained_gates = set()

    def get_gate_data(self, gate_name):
        if gate_name == "AND":
            return self.AND_inputs, self.AND_labels
        elif gate_name == "OR":
            return self.OR_inputs, self.OR_labels
        elif gate_name == "NAND":
            return self.NAND_inputs, self.NAND_labels
        elif gate_name == "NOR":
            return self.NOR_inputs, self.NOR_labels
        elif gate_name == "XOR":
            return self.XOR_inputs, self.XOR_labels
        else:
            raise ValueError(f"Compuerta {gate_name} no reconocida")
            
    def get_expected_output(self, gate_name, input1, input2):
        if gate_name == "AND":
            return 1 if input1 == 1 and input2 == 1 else 0
        elif gate_name == "OR":
            return 0 if input1 == 0 and input2 == 0 else 1
        elif gate_name == "NAND":
            return 0 if input1 == 1 and input2 == 1 else 1
        elif gate_name == "NOR":
            return 1 if input1 == 0 and input2 == 0 else 0
        elif gate_name == "XOR":
            return 1 if input1 != input2 else 0
        else:
            raise ValueError(f"Compuerta {gate_name} no reconocida")
            
    def mark_as_trained(self, gate_name):
        self.trained_gates.add(gate_name)
    
    def unmark_as_trained(self, gate_name):
        """Marca una compuerta como no entrenada"""
        if gate_name in self.trained_gates:
            self.trained_gates.remove(gate_name)

    def is_trained(self, gate_name):
        """
        Verifica si una compuerta ha sido entrenada.

        Args:
            gate_name (str): El nombre de la compuerta a verificar.

        Returns:
            bool: True si la compuerta ha sido entrenada, False en caso contrario.
        """
        return gate_name in self.trained_gates