# Implementation of single layer perceptron from scratch with different loss functions.

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Loss functions.
def perceptron_loss(y, g_x):
    return (y * g_x <= 0).astype(float)

def squared_error_loss(y, g_x):
    return (y - g_x) ** 2

def binary_cross_entropy_loss(y, g_x):
    if y > 0:
        return np.log(1 + np.exp(-g_x))
    else:
        return np.log(1 + np.exp(g_x))

def hinge_loss(y, g_x):
    return np.maximum(0, 1 - y * g_x)

# Gradient Descent Perceptron Training
def train_perceptron(X, y, loss_fn, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_samples):
            g_x = np.dot(X[i], w) + b
            loss = loss_fn(y[i], g_x)
            total_loss += loss
            
            # Update weights and bias using the gradient of the loss
            if loss_fn == perceptron_loss:
                if y[i] * g_x <= 0:  # Misclassified
                    w += lr * y[i] * X[i]
                    b += lr * y[i]
            elif loss_fn == squared_error_loss:
                gradient = -2 * (y[i] - g_x) * X[i]
                w -= lr * gradient
                b -= lr * -2 * (y[i] - g_x)
            elif loss_fn == binary_cross_entropy_loss:
                gradient = (sigmoid(-y[i] * g_x) - 1) * y[i] * X[i]
                w -= lr * gradient
                b -= lr * (sigmoid(-y[i] * g_x) - 1) * y[i]
            elif loss_fn == hinge_loss:
                if 1 - y[i] * g_x > 0:
                    w += lr * y[i] * X[i]
                    b += lr * y[i]
        
        losses.append(total_loss / n_samples)  # Average loss over the dataset

    return w, b, losses

# Generate an augmented dataset
def generate_dataset():
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)  # Label: 1 if x1 + x2 > 0, else -1
    return X, y

# Plot decision boundary
def plot_decision_boundary(X, y, w, b, title="Decision Boundary"):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    Z = np.dot(np.c_[xx1.ravel(), xx2.ravel()], w) + b
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, levels=[-1, 0, 1], alpha=0.7, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Generate dataset
    X, y = generate_dataset()

    # Split the dataset
    n = len(X)
    train_X, val_X, test_X = X[:int(0.6*n)], X[int(0.6*n):int(0.8*n)], X[int(0.8*n):]
    train_y, val_y, test_y = y[:int(0.6*n)], y[int(0.6*n):int(0.8*n)], y[int(0.8*n):]

    # Train the perceptron for each loss function
    for loss_name, loss_fn in [
        ("Perceptron Loss", perceptron_loss),
        ("Squared Error Loss", squared_error_loss),
        ("Binary Cross-Entropy Loss", binary_cross_entropy_loss),
        ("Hinge Loss", hinge_loss)
    ]:
        print(f"Training with {loss_name}...")
        w, b, losses = train_perceptron(train_X, train_y, loss_fn, lr=0.1, epochs=50)
        
        # Plot decision boundary
        plot_decision_boundary(train_X, train_y, w, b, title=f"{loss_name} Decision Boundary")

        # Print final weights and bias
        print(f"Weights: {w}, Bias: {b}")

        # Evaluate on validation and test data
        val_accuracy = np.mean(np.sign(np.dot(val_X, w) + b) == val_y)
        test_accuracy = np.mean(np.sign(np.dot(test_X, w) + b) == test_y)
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
