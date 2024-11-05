import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Cargar y preparar los datos
file_path = r'C:\Users\Admin\Documents\IA\Q2\BaseDatosFacturaElectronica.csv'
new_data = pd.read_csv(file_path)
new_data['documentoFecha'] = pd.to_datetime(new_data['documentoFecha'], errors='coerce')
new_data['dia'] = new_data['documentoFecha'].dt.day

# Crear variable binaria para "Documento Rechazado"
new_data['rechazado'] = new_data['documentoEstadoDIAN'].apply(lambda x: 1 if x == 'Documento Rechazado' else 0)

# Agrupar rechazos diarios por tipo de documento
daily_rejections = new_data[new_data['rechazado'] == 1].groupby(['dia', 'documentoTipo']).size().unstack(fill_value=0)
daily_rejections.reset_index(inplace=True)

# Seleccionar 'dia' como característica y rechazos del tipo 92 como objetivo
X = daily_rejections[['dia']].values  # Característica: día del mes
y = daily_rejections[92].values       # Objetivo: número de rechazos para documento tipo 92

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Entrenar el modelo de regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_r2 = r2_score(y_test, y_pred)

# Generar la línea de regresión para visualización
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = lin_reg.predict(X_range)

# Imprimir los resultados del análisis
print("Error Cuadrático Medio (MSE):", lin_mse)
print("Coeficiente de Determinación (R²):", lin_r2)

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Datos Reales')
plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Línea de Regresión')
plt.xlabel('Día del Mes')
plt.ylabel('Rechazos Documentos Tipo 92')
plt.title('Regresión Lineal - Predicción de Rechazos Documentos Tipo 92')
plt.legend()
plt.grid(True)
plt.show()
