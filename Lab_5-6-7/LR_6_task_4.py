import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Завантаження набору даних про діабет
diabetes = load_diabetes()
X = diabetes.data  # Матриця ознак
y = diabetes.target  # Цільова змінна

# Розбивка даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання моделі лінійної регресії
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогнозування для тестової вибірки
y_pred = regr.predict(X_test)
# Побудова графіка
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), label='Передбачення')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ідеальна залежність')
ax.set_xlabel('Виміряно (y_test)', fontsize=12)
ax.set_ylabel('Передбачено (y_pred)', fontsize=12)
ax.set_title('Фактичні vs Передбачені значення', fontsize=14)
ax.legend()
plt.grid(True)
plt.show()
# Розрахунок коефіцієнтів регресії та метрик
coefficients = regr.coef_  # Коефіцієнти регресії
intercept = regr.intercept_  # Вільний член
mae = mean_absolute_error(y_test, y_pred)  # Середня абсолютна помилка
mse = mean_squared_error(y_test, y_pred)  # Середньоквадратична помилка
r2 = r2_score(y_test, y_pred)  # R²
# Виведення результатів
print("Коефіцієнти регресії:", coefficients)
print("Вільний член (intercept):", intercept)
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("R^2 Score:", round(r2, 2))
