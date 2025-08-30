import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.history = []  # To save history

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

#This line will save the copy of the weights for visualization
            self.history.append((self.weights.copy(), self.bias))

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


#Just a Random Dataset (Matrix)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

#Using the model
model = MyLinearRegression(learning_rate=0.1, n_iters=20)
model.fit(X, y)


plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="red", label="Data points")

x_range = np.linspace(0, 6, 100)
for i, (w, b) in enumerate(model.history):
    if i % 2 == 0:  # plot every 2 steps so it's not cluttered
        y_line = w * x_range + b
        plt.plot(x_range, y_line, alpha=0.5, label=f"Iter {i}")

final_y = model.weights * x_range + model.bias
plt.plot(x_range, final_y, color="blue", linewidth=2, label="Final line")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Gradient Descent Convergence of Linear Regression")
plt.legend()
plt.show()
