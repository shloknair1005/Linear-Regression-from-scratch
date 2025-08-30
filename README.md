# Linear Regression from Scratch

This project implements **Linear Regression** completely from scratch using Python and NumPy — without relying on machine learning libraries like scikit-learn. It helps understand the underlying mathematics, gradient descent optimization, and how models actually learn.

---

## 📌 Project Overview
Linear Regression is one of the simplest machine learning algorithms. It models the relationship between a **dependent variable (y)** and an **independent variable (x)** by fitting a straight line.

The equation is:

\[
y = w \cdot x + b
\]

- **w** → weight (slope of the line)  
- **b** → bias (intercept)  
- **x** → input feature  
- **y** → predicted output  

The model learns `w` and `b` using **gradient descent** to minimize the **Mean Squared Error (MSE)** between predicted and actual values.

---

## 🚀 Features
- Linear Regression implemented from scratch  
- Gradient Descent optimization  
- Loss function tracking (MSE)  
- Visualization of regression line and error reduction  

---

## 📊 Visualization
- Scatter plot of data points  
- Best-fit regression line  
- Loss vs Epochs curve to show model convergence  

---

## 🛠️ How It Works
1. Initialize weights and bias  
2. Predict outputs using the linear equation  
3. Compute the loss (MSE)  
4. Update weights & bias using gradient descent  
5. Repeat until convergence  

---

## 🔮 Future Improvements
- Extend to **Multiple Linear Regression**  
- Add **Polynomial Regression**  
- Support **Stochastic Gradient Descent (SGD)**  
- Compare with **scikit-learn’s LinearRegression**  
- Add interactive **Jupyter Notebook demo**  

---

## 📌 Learning Outcomes
- Understand the math behind linear regression  
- Learn how gradient descent updates parameters  
- Gain intuition about optimization in ML models  
