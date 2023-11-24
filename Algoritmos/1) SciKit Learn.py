# Estimators - entrenando
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Creación del clasificador
clf = RandomForestClassifier(random_state=0)

# Datos de entrenamiento
X = [[1, 2, 3], [11, 12, 13]]
y = [0, 1]

# Entrenamiento del clasificador
clf.fit(X, y)

# Estimators - prediciendo
# Predicciones en los datos de entrenamiento
predictions_train = clf.predict(X)
print("Predicciones en datos de entrenamiento:")
print(predictions_train)

# Predicciones en nuevos datos
new_data = [[4, 5, 6], [14, 15, 16]]
predictions_new_data = clf.predict(new_data)
print("\nPredicciones en nuevos datos:")
print(predictions_new_data)

# =======================================================================
# Transformatos - entrenando

# Datos
Z = [[0, 15], [1, -10]]

# Crear una instancia del escalador
scaler = StandardScaler()

# Ajustar el escalador a los datos
scaler.fit(Z)

# Imprimir la media y la desviación estándar aprendidas durante el ajuste
print("Media:", scaler.mean_)
print("Desviación Estándar:", scaler.scale_)

# Transformar nuevos datos
Z_transformed = scaler.transform(Z)
print("\nDatos transformados:")
print(Z_transformed)
