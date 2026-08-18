"""
پروژه: پیاده‌سازی شبکه عصبی (MLP) از صفر با NumPy روی دیتاست MNIST
هدف: طبقه‌بندی ارقام دست‌نویس، بدون استفاده از PyTorch/TensorFlow
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# ============================================================
# ۱. بارگذاری و آماده‌سازی داده
# ============================================================
print("در حال بارگذاری MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

X = X / 255.0  # نرمالایز پیکسل‌ها بین 0 و 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def one_hot(y, num_classes=10):
    """تبدیل برچسب عددی به بردار one-hot"""
    return np.eye(num_classes)[y]


y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)


# ============================================================
# ۲. مقداردهی اولیه شبکه
# ============================================================
input_size = 784
hidden_size = 64
output_size = 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


# ============================================================
# ۳. توابع فعال‌سازی
# ============================================================
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================
# ۴. Forward Pass
# ============================================================
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


# ============================================================
# ۵. محاسبه Loss (Cross-Entropy)
# ============================================================
def compute_loss(y_true_oh, y_pred):
    m = y_true_oh.shape[0]
    return -np.sum(y_true_oh * np.log(y_pred + 1e-9)) / m


# ============================================================
# ۶. Backpropagation
# ============================================================
def backward(X, y_true_oh, z1, a1, z2, a2, W2):
    m = X.shape[0]

    dz2 = a2 - y_true_oh
    dW2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T @ dz1 / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# ============================================================
# ۷. حلقه آموزش (Gradient Descent)
# ============================================================
learning_rate = 0.1
epochs = 100
losses = []

for epoch in range(epochs):
    z1, a1, z2, a2 = forward(X_train, W1, b1, W2, b2)
    loss = compute_loss(y_train_oh, a2)
    losses.append(loss)

    dW1, db1, dW2, db2 = backward(X_train, y_train_oh, z1, a1, z2, a2, W2)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# ============================================================
# ۸. ارزیابی روی داده تست
# ============================================================
_, _, _, test_pred = forward(X_test, W1, b1, W2, b2)
accuracy_mlp = np.mean(np.argmax(test_pred, axis=1) == y_test)
print(f"دقت MLP از صفر: {accuracy_mlp:.4f}")


# ============================================================
# ۹. مقایسه با Logistic Regression (sklearn)
# ============================================================
print("در حال آموزش Logistic Regression برای مقایسه...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
accuracy_logreg = log_reg.score(X_test, y_test)
print(f"دقت Logistic Regression: {accuracy_logreg:.4f}")


# ============================================================
# ۱۰. رسم نمودار Loss
# ============================================================
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("منحنی کاهش Loss در طول آموزش MLP")
plt.savefig("loss_curve.png")
print("نمودار در loss_curve.png ذخیره شد.")
