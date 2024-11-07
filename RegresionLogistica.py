import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Cargar los datos y prepararlos (como en el código anterior)
file_path = r'C:\Archivos\Expanded_BaseDatosProductivoFacturaElectronica3.csv'
data = pd.read_csv(file_path)
data['documentoFecha'] = pd.to_datetime(data['documentoFecha'], errors='coerce')
data_october = data[(data['documentoFecha'].dt.year == 2024) & (data['documentoFecha'].dt.month == 10)]
data_october['dia'] = data_october['documentoFecha'].dt.day
data_october['rechazado'] = data_october['documentoEstadoDIAN'].apply(lambda x: 1 if x == 'Documento Rechazado' else 0)

# Entrenar el modelo de regresión logística
X_rejection = data_october[['dia', 'documentoTipo']].copy()
y_rejection = data_october['rechazado']
X_train, X_test, y_train, y_test = train_test_split(X_rejection, y_rejection, test_size=0.3, random_state=0)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Crear proyecciones para el día 29
doc_types = data_october['documentoTipo'].unique()
X_pred = pd.DataFrame({'dia': [29] * len(doc_types), 'documentoTipo': doc_types})
predictions_29 = log_reg.predict_proba(X_pred)[:, 1]
average_daily_docs_by_type = data_october.groupby('documentoTipo')['dia'].count() / len(data_october['dia'].unique())
proj_rejections_29 = average_daily_docs_by_type[doc_types] * predictions_29

# Crear un DataFrame para visualización
projection_df = pd.DataFrame({
    'documentoTipo': doc_types,
    'rechazos_proyectados_29': proj_rejections_29
})

# Convertir 'documentoTipo' a una categoría ordenada para mejorar la distribución
projection_df['documentoTipo'] = projection_df['documentoTipo'].astype(str)
projection_df = projection_df.sort_values('rechazos_proyectados_29', ascending=False)  # Ordenar para mejor visualización

# Graficar con seaborn para mejorar la distribución de las barras
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='documentoTipo', y='rechazos_proyectados_29', data=projection_df, color='steelblue')
ax.set_xlabel('Tipo de Documento')
ax.set_ylabel('Proyección de Documentos Rechazados')
ax.set_title('Proyección de Documentos Rechazados por Tipo para Octubre 29')

# Añadir etiquetas encima de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()