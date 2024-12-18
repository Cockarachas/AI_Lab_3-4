from sklearn.preprocessing import PolynomialFeatures
import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm

# Вхідний файл, який містить дані
input_file = 'data_multivar_regr.txt'
# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Розбивка даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]
# Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
# Прогнозування результату для тестових даних
y_test_pred_linear = linear_regressor.predict(X_test)
# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=3)  # Змінюйте degree для експериментів
X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.transform(X_test)

# Модель поліноміальної регресії
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_train_poly, y_train)
# Прогнозування результату для тестових даних (поліноміальна регресія)
y_test_pred_poly = poly_regressor.predict(X_test_poly)
# Оцінка продуктивності
print("Linear Regressor Performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_linear), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_linear), 2))

print("\nPolynomial Regressor Performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_poly), 2))

# Прогноз для нового датапойнта
datapoint = [[7.75, 6.35, 5.56]]  # Змінюйте для перевірки
datapoint_linear = linear_regressor.predict(datapoint)
datapoint_poly = poly_regressor.predict(polynomial.transform(datapoint))

print("\nPrediction for datapoint:")
print("Linear regression prediction:", datapoint_linear)
print("Polynomial regression prediction:", datapoint_poly)

# Збереження моделі лінійної регресії
output_model_file = '../model_linear.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

# Завантаження моделі
with open(output_model_file, 'rb') as f:
    loaded_model = pickle.load(f)
y_test_pred_new = loaded_model.predict(X_test)

print("\nNew mean absolute error for loaded linear model:",
      round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
