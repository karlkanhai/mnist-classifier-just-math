import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Fetching the data
def fetch_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target.astype(int), test_size=0.3, random_state=42)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = fetch_mnist_data()

# Splitting the data into training and validation sets
X_val, Y_val = X_train[50000:], Y_train[50000:]
X_train, Y_train = X_train[:50000], Y_train[:50000]

# Initializing weights
def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size)

l1 = initialize_weights(784, 128)
l2 = initialize_weights(128, 10)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def softmax_derivative(x):
    return x * (1 - x)

# Forward and backward pass
def forward_backward(x, y, l1, l2):
    x_sigmoid = sigmoid(np.dot(x, l1))
    out = softmax(np.dot(x_sigmoid, l2))
    
    targets = np.zeros_like(out)
    targets[np.arange(len(y)), y] = 1
    
    error = 2 * (out - targets)
    d_out = error * softmax_derivative(out)
    update_l2 = np.dot(x_sigmoid.T, d_out)
    
    d_x_sigmoid = np.dot(d_out, l2.T) * sigmoid_derivative(x_sigmoid)
    update_l1 = np.dot(x.T, d_x_sigmoid)
    
    return out, update_l1, update_l2

# Calculate accuracy
def calculate_accuracy(predictions, labels):
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == labels)
    return accuracy

# Calculate loss using mean squared error
def calculate_loss(predictions, labels):
    targets = np.zeros_like(predictions)
    targets[np.arange(len(labels)), labels] = 1
    loss = np.mean(np.sum((predictions - targets) ** 2, axis=1))
    return loss

# Training the model using SGD
learning_rate = 0.001
batch_size = 128
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    total_accuracy = 0
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = Y_train[i:i+batch_size]
        
        out, update_l1, update_l2 = forward_backward(x_batch, y_batch, l1, l2)
        
        l1 -= learning_rate * update_l1
        l2 -= learning_rate * update_l2
        
        batch_loss = calculate_loss(out, y_batch)
        batch_accuracy = calculate_accuracy(out, y_batch)
        
        total_loss += batch_loss
        total_accuracy += batch_accuracy

    average_loss = total_loss / (len(X_train) / batch_size)
    average_accuracy = total_accuracy / (len(X_train) / batch_size)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}")

# Testing the model on test data
test_predictions = forward_backward(X_test, Y_test, l1, l2)[0]
test_accuracy = calculate_accuracy(test_predictions, Y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
