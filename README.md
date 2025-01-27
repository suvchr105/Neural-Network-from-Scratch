# Custom Neural Network Implementation for MNIST Classification

A flexible and modular implementation of a neural network from scratch using NumPy, designed for the MNIST digit classification task. This implementation features various activation functions, weight initialization methods, and training optimizations including early stopping and L2 regularization.

## 🌟 Features

- **Multiple Activation Functions**:
  - ReLU (Rectified Linear Unit)
  - Tanh (Hyperbolic Tangent)
  - Leaky ReLU
  - Softmax (output layer)

- **Advanced Implementation Details**:
  - Xavier and He weight initialization strategies
  - Mini-batch gradient descent
  - L2 regularization for preventing overfitting
  - Early stopping mechanism
  - Categorical cross-entropy loss function

- **Comprehensive Analysis Tools**:
  - Training history visualization
  - Confusion matrix plotting
  - Detailed performance metrics
  - Model parameter analysis
  - Cross-validation with different train-test splits

## 📊 Project Structure

```
neural_network/
├── Layer.py                 # Base layer implementation
├── Activations.py          # Various activation functions
├── Loss.py                 # Loss function implementation
├── NeuralNetwork.py        # Main neural network class
└── utils/
    ├── data_processing.py  # Data loading and preprocessing
    └── visualization.py    # Plotting and visualization tools
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy sklearn matplotlib seaborn pandas
```

### Usage

```python
from neural_network import NeuralNetwork

# Initialize the network
network = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    seed=32,
    activation='relu',
    initialization='he'
)

# Train the model
history = network.train(
    X_train, 
    y_train,
    batch_size=24,
    epochs=25,
    validation_split=0.2,
    early_stopping_patience=5,
    lambda_reg=0.01
)

# Evaluate the model
metrics = network.evaluate(X_test, y_test)
```

## 📈 Model Architecture

- Input Layer: 784 neurons (28x28 MNIST images)
- Hidden Layer 1: 128 neurons
- Hidden Layer 2: 64 neurons
- Output Layer: 10 neurons (digit classes 0-9)

## 🔍 Experimental Results

The implementation includes comprehensive experiments with:
- Different train-test splits (70:30, 80:20, 90:10)
- Various activation functions (ReLU, Tanh, Leaky ReLU)
- Different initialization strategies (Xavier, He)

Results are automatically saved to `neural_network_results.csv` and include:
- Training and validation metrics
- Test accuracy and loss
- Training time
- Model parameters count
- Precision, recall, and F1 scores

## 📊 Visualization

The implementation provides various visualization tools:
- Training history plots (loss and accuracy)
- Confusion matrices
- Performance comparison plots across different configurations
- Parameter distribution analysis

## 🔧 Customization

You can easily customize the network by modifying:
- Layer sizes
- Activation functions
- Initialization methods
- Training parameters (batch size, learning rate, etc.)
- Regularization strength
- Early stopping patience

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

If you have any questions or would like to contribute to this project, please open an issue or submit a pull request.

---
Created with ❤️ by Suvadip Chakraborty
