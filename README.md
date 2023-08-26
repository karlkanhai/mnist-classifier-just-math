# MNIST Classifier from Scratch

No fancy frameworks, just pure python and numpy. This project was done to deeper understand neural networks by building one from the ground up.

## How It Works

1. **Fetching the Data**: The MNIST dataset is fetched directly from its source. The dataset is then split into training and testing sets. The training set is further divided into training and validation subsets.

2. **Initializing Weights**: The neural network consists of three layers: an input layer with 784 units (corresponding to the 28x28 pixel images), a hidden layer with 128 units, and an output layer with 10 units (representing digits 0-9). Weights are initialized with random values from a normal distribution.

3. **Activation Functions**: 
   - **Sigmoid**: Used for the hidden layer.
   - **Softmax**: Applied to the output layer.
4. **Forward and Backward Pass**: 
   - **Forward Pass**: Input data is passed through the network to generate predictions using the formula:
     a = sigma(W * x + b)
     where W is the weight matrix, x is the input, b is the bias, and a is the activation.
   - **Backward Pass**: The difference between predictions and actual labels (error) is used to compute gradients and update the network's weights. The gradient descent update rule is:
     W = W - alpha * gradient_W J
     where alpha is the learning rate and gradient_W J is the gradient of the loss function J with respect to W.

5. **Training with SGD**: The model is trained using Stochastic Gradient Descent (SGD). The training data is processed in batches, and after each batch, the weights are updated to minimize the error. SGD helps in faster convergence and escaping local minima.

6. **Evaluation**: After training, the model's performance is evaluated on a validation set to check its accuracy. This helps in understanding how well the model will perform on unseen data.


## Setup

**Requirements**:
- Python
- NumPy
- scikit-learn

**Steps**:
1. Clone the repo:
```bash
   git clone https://github.com/your_username/simple-mnist-classifier.git
```
2. Install the dependencies:
```bash
pip install numpy scikit-learn
```
**Running**:
```bash
python mnist.py
```

## Credits

MNIST dataset
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

