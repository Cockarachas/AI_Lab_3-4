import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Генерація випадкових даних
m = 100
X = 6 * np.random.rand(m, 1) - 4  # Генерація випадкових значень X в межах [-4, 2]
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)  # Формула для y з шумом

# Побудова графіка для даних
plt.scatter(X, y, color='blue', label='Дані')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Генерація випадкових даних')
plt.legend()
plt.show()

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Поліноміальна регресія
poly = PolynomialFeatures(degree=2)  # Ступінь полінома 2
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Побудова графіків
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred_lin, color='red', label='Лінійна регресія')
plt.plot(X, y_pred_poly, color='green', label='Поліноміальна регресія (ст. 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна та поліноміальна регресія')
plt.legend()
plt.show()

# Оцінка якості моделей
mae_lin = mean_absolute_error(y, y_pred_lin)
mse_lin = mean_squared_error(y, y_pred_lin)
r2_lin = r2_score(y, y_pred_lin)

mae_poly = mean_absolute_error(y, y_pred_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

# Виведення результатів
print("Linear regressor performance:")
print(f"Mean absolute error = {mae_lin:.2f}")
print(f"Mean squared error = {mse_lin:.2f}")
print(f"R2 score = {r2_lin:.2f}")

print("\nPolynomial Regressor Performance:")
print(f"Mean absolute error = {mae_poly:.2f}")
print(f"Mean squared error = {mse_poly:.2f}")
print(f"R2 score = {r2_poly:.2f}")
