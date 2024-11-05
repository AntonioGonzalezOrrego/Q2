import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# Cargar y preparar los datos (asegúrate de cargar tu archivo CSV correctamente)
file_path = r'C:\Users\Admin\Documents\IA\Q2\BaseDatosFacturaElectronica.csv'
new_data = pd.read_csv(file_path)
new_data['documentoFecha'] = pd.to_datetime(new_data['documentoFecha'], errors='coerce')
new_data['dia'] = new_data['documentoFecha'].dt.day

# Crear variable binaria para "Documento Rechazado"
new_data['rechazado'] = new_data['documentoEstadoDIAN'].apply(lambda x: 1 if x == 'Documento Rechazado' else 0)

# Agrupar rechazos diarios por tipo de documento
daily_rejections = new_data[new_data['rechazado'] == 1].groupby(['dia', 'documentoTipo']).size().unstack(fill_value=0)
daily_rejections.reset_index(inplace=True)

# Escalar los datos para mejorar el rendimiento de la red neuronal
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(daily_rejections[['dia']])  # Escalar 'dia'
y_scaled = scaler.fit_transform(daily_rejections[[92]])  # Escalar rechazos para tipo 92

# Separar en conjunto de entrenamiento y prueba
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=0)

# Definir el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),              # Nueva forma de definir el input
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train_nn, y_train_nn, epochs=200, validation_split=0.2, verbose=0)

# Realizar predicciones
y_pred_nn = model.predict(X_test_nn)

# Escalar de regreso las predicciones para interpretar los resultados
y_pred_nn_rescaled = scaler.inverse_transform(y_pred_nn)
y_test_nn_rescaled = scaler.inverse_transform(y_test_nn)

# Calcular el MSE y el R² para evaluar el modelo
nn_mse = mean_squared_error(y_test_nn_rescaled, y_pred_nn_rescaled)
nn_r2 = r2_score(y_test_nn_rescaled, y_pred_nn_rescaled)

print("MSE de la Red Neuronal:", nn_mse)
print("R² de la Red Neuronal:", nn_r2)

# Visualización del desempeño del modelo
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Historial de Pérdida en el Entrenamiento de la Red Neuronal')
plt.show()

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(scaler.inverse_transform(X_test_nn), y_test_nn_rescaled, color='blue', label='Datos Reales')
plt.scatter(scaler.inverse_transform(X_test_nn), y_pred_nn_rescaled, color='red', label='Predicciones')
plt.xlabel('Día del Mes')
plt.ylabel('Rechazos Documentos Tipo 92')
plt.legend()
plt.title('Red Neuronal - Predicción de Rechazos Documentos Tipo 92')
plt.show()
