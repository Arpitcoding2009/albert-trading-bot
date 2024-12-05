from typing import List, Dict, Optional
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

class QuantumCircuit:
    """Quantum circuit for financial data processing and prediction"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Create quantum circuit
        self.circuit = qml.QNode(self._circuit, self.dev)
        
        # Initialize parameters
        self.params = self._init_parameters()

    def _init_parameters(self) -> np.ndarray:
        """Initialize circuit parameters"""
        # Parameters for rotation gates
        n_rotations = self.n_qubits * self.n_layers * 3  # Rx, Ry, Rz for each qubit
        # Parameters for entangling gates
        n_entangling = self.n_qubits * (self.n_layers - 1)
        
        return np.random.uniform(low=-np.pi, high=np.pi, 
                               size=(n_rotations + n_entangling,))

    def _circuit(self, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Define the quantum circuit architecture"""
        # Encode classical data into quantum state
        self._encode_input(inputs)
        
        param_idx = 0
        # Apply parametrized gates in layers
        for layer in range(self.n_layers):
            # Rotation gates
            for qubit in range(self.n_qubits):
                qml.Rot(params[param_idx], 
                       params[param_idx + 1],
                       params[param_idx + 2], 
                       wires=qubit)
                param_idx += 3
            
            # Entangling gates (except for last layer)
            if layer < self.n_layers - 1:
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                    param_idx += 1
        
        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _encode_input(self, inputs: np.ndarray):
        """Encode classical data into quantum state"""
        # Amplitude encoding
        for i in range(min(len(inputs), self.n_qubits)):
            qml.RY(inputs[i], wires=i)

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess classical data for quantum circuit"""
        # Normalize data to [-π, π]
        return 2 * np.pi * (data - np.min(data)) / (np.max(data) - np.min(data)) - np.pi

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction using quantum circuit"""
        # Preprocess features
        processed_features = self._preprocess_data(features)
        
        # Execute quantum circuit
        prediction = self.circuit(processed_features, self.params)
        
        # Post-process quantum output
        return self._postprocess_prediction(prediction)

    def _postprocess_prediction(self, quantum_output: np.ndarray) -> np.ndarray:
        """Post-process quantum circuit output"""
        # Convert from [-1, 1] to [0, 1]
        return (quantum_output + 1) / 2

    def train(self, X: np.ndarray, y: np.ndarray, 
              n_epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.01) -> Dict[str, List[float]]:
        """Train the quantum circuit"""
        opt = qml.GradientDescentOptimizer(learning_rate)
        
        batch_size = min(batch_size, len(X))
        n_batches = len(X) // batch_size
        
        loss_history = []
        accuracy_history = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Update parameters
                self.params = opt.step(
                    lambda p: self._loss_function(p, X_batch, y_batch),
                    self.params
                )
                
                # Calculate metrics
                batch_loss = self._loss_function(self.params, X_batch, y_batch)
                batch_acc = self._calculate_accuracy(X_batch, y_batch)
                
                epoch_loss += batch_loss
                epoch_acc += batch_acc
            
            # Average metrics
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            loss_history.append(epoch_loss)
            accuracy_history.append(epoch_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")
        
        return {
            'loss_history': loss_history,
            'accuracy_history': accuracy_history
        }

    def _loss_function(self, params: np.ndarray, X: np.ndarray, 
                      y: np.ndarray) -> float:
        """Calculate loss for training"""
        predictions = np.array([self.circuit(x, params) for x in X])
        return np.mean((predictions - y) ** 2)

    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy"""
        predictions = np.array([self.predict(x) for x in X])
        return np.mean(np.round(predictions) == y)

    def save_params(self, filepath: str):
        """Save circuit parameters"""
        np.save(filepath, self.params)

    def load_params(self, filepath: str):
        """Load circuit parameters"""
        self.params = np.load(filepath)
