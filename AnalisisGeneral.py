import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar los datos
file_path = r'C:\Archivos\Expanded_BaseDatosProductivoFacturaElectronica3.csv'
data = pd.read_csv(file_path)

# Convertir 'documentoFecha' a datetime
data['documentoFecha'] = pd.to_datetime(data['documentoFecha'], errors='coerce')

# Filtrar registros de octubre 2024
data_october = data[(data['documentoFecha'].dt.year == 2024) & (data['documentoFecha'].dt.month == 10)]

# 1. Distribución de Documentos por Tipo
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data_october, x='documentoTipo')
ax.set_title('Distribución de Documentos por Tipo en Octubre 2024')
ax.set_xlabel('Tipo de Documento')
ax.set_ylabel('Cantidad')
for p in ax.patches:
    height = p.get_height()
    if not pd.isna(height):
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Distribución de Documentos por Estado
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data_october, x='documentoEstadoDIAN')
ax.set_title('Distribución de Documentos por Estado en Octubre 2024')
ax.set_xlabel('Estado del Documento')
ax.set_ylabel('Cantidad')
for p in ax.patches:
    height = p.get_height()
    if not pd.isna(height):
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Top 10 Empresas con Más Documentos Procesados
top_companies_total = data_october['empresaNombre'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=top_companies_total.index, y=top_companies_total.values)
ax.set_title('Top 10 Empresas con Más Documentos Procesados en Octubre 2024')
ax.set_xlabel('Empresa')
ax.set_ylabel('Cantidad de Documentos Procesados')
for p in ax.patches:
    height = p.get_height()
    if not pd.isna(height):
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 4. Top 10 Empresas con Más Documentos en Estado "Documento Rechazado"
top_documento_rechazado_companies = data_october[data_october['documentoEstadoDIAN'] == 'Documento Rechazado']['empresaNombre'] \
    .value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=top_documento_rechazado_companies.index, y=top_documento_rechazado_companies.values)
ax.set_title('Top 10 Empresas con Más Documentos en Estado "Documento Rechazado" en Octubre 2024')
ax.set_xlabel('Empresa')
ax.set_ylabel('Doc. En Estado "Documento Rechazado"')
for p in ax.patches:
    height = p.get_height()
    if not pd.isna(height):
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 6. Análisis Temporal de Emisión de Documentos por Día
daily_counts = data_october['documentoFecha'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
daily_counts.plot(kind='line', marker='o')
plt.title('Emisión Diaria de Documentos en Octubre 2024')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de Documentos Emitidos')
plt.grid()
plt.tight_layout()
plt.show()

# 7. Regresión Lineal para Proyección de Documentos Procesados en los Días 29, 30 y 31 de Octubre
# Preparar los datos para el modelo de regresión
daily_document_counts = daily_counts.reset_index()
daily_document_counts.columns = ['documentoFecha', 'documentCount']
daily_document_counts['day_of_month'] = daily_document_counts['documentoFecha'].dt.day

# Definir X e y para la regresión
X = daily_document_counts['day_of_month'].values.reshape(-1, 1)
y = daily_document_counts['documentCount'].values

# Entrenar el modelo de regresión lineal
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Predecir la cantidad de documentos para los días 29, 30 y 31 de octubre
future_days = np.array([29, 30, 31]).reshape(-1, 1)
predictions = linear_reg.predict(future_days)

# Graficar los datos reales y la proyección
plt.figure(figsize=(10, 6))
plt.plot(daily_document_counts['day_of_month'], daily_document_counts['documentCount'], marker='o', label='Real')
plt.plot(
    np.concatenate([daily_document_counts['day_of_month'], future_days.ravel()]),
    np.concatenate([y, predictions]), 
    marker='x', linestyle='--', color='red', label='Proyectado'
)
plt.title('Proyección del Flujo de Documentos Procesados en Octubre 2024')
plt.xlabel('Día de Octubre')
plt.ylabel('Cantidad de Documentos Procesados')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Imprimir las proyecciones
print("Proyección de documentos procesados para los días 29, 30 y 31 de octubre:")
for day, count in zip([29, 30, 31], predictions):
    print(f"Día {day}: {int(count)} documentos")