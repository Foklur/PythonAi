import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


np.random.seed(42)
times = np.random.randint(0, 1440, 5000)
durations = 20 + 10 * np.sin(2 * np.pi * times / 1440) + np.random.normal(0, 2, 5000)

x_sin = np.sin(2 * np.pi * times / 1440)
x_cos = np.cos(2 * np.pi * times / 1440)
X = np.column_stack((x_sin, x_cos))
y = durations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=1000)
model.fit(X_train, y_train)

def transform_time(hour, minute):
    minutes = hour * 60 + minute
    return [np.sin(2 * np.pi * minutes / 1440), np.cos(2 * np.pi * minutes / 1440)]

predict_times = [(10, 30), (0, 0), (2, 40)]
X_pred = np.array([transform_time(h, m) for h, m in predict_times])
predicted_durations = model.predict(X_pred)

for (h, m), duration in zip(predict_times, predicted_durations):
    print(f"Час: {h:02}:{m:02} → прогноз тривалості :: {duration:.2f} хв")

plt.figure(figsize=(10, 4))
plt.scatter(times, durations, s=2, alpha=0.3, label='Справжні')
minutes_range = np.arange(0, 1440)
X_full = np.column_stack([
    np.sin(2 * np.pi * minutes_range / 1440),
    np.cos(2 * np.pi * minutes_range / 1440)
])
y_pred_full = model.predict(X_full)
plt.plot(minutes_range, y_pred_full, color='red', label='NN передбачення')
plt.xlabel('Час у хвилинах')
plt.ylabel('Тривалість (хв)')
plt.title('Передбачення тривалості поїздки (MLPRegressor)')
plt.grid(True)
plt.legend()
plt.show()
