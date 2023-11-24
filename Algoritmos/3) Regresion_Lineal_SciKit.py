import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

reg = LinearRegression().fit(X, y)

# Imprimir coeficientes (pendientes) de la regresión
print("Coeficientes (pendientes):", reg.coef_)

# Imprimir término independiente de la regresión
print("Término independiente:", reg.intercept_)

# Hacer una predicción para nuevos datos
new_data = np.array([[3, 5]])
predictions = reg.predict(new_data)
print("Predicciones para nuevos datos:", predictions)
