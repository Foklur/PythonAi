import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X) + 0.1 * X**2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', max_iter=5000, random_state=42)
model.fit(X_train, y_train.ravel()) 
y_predict = model.predict(X_test)
y_all_predict = model.predict(X)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print(f"MAE :: {mae:.4f}")
print(f"MSE :: {mse:.4f}")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(X, y, label='Реальна ф-ція', color='blue')
plt.plot(X, y_all_predict, label='Прогноз', color='red', linestyle='--')
plt.title('Реальна vs Передбачена')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X, np.abs(y.ravel() - y_all_predict), label='Абсолютна помилка', color='purple')
plt.title('Абсолютна помилка прогнозу')
plt.xlabel('x')
plt.ylabel('|y - y_predict|')
plt.legend()

plt.tight_layout()
plt.show()
