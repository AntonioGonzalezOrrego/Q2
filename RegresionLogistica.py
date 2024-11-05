import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Cargar y preparar los datos
file_path = r'C:\Users\Admin\Documents\IA\Q2\BaseDatosFacturaElectronica.csv'
new_data = pd.read_csv(file_path)
new_data['documentoFecha'] = pd.to_datetime(new_data['documentoFecha'], errors='coerce')
new_data['dia'] = new_data['documentoFecha'].dt.day

# Crear variable objetivo binaria para "Documento Rechazado"
new_data['rechazado'] = new_data['documentoEstadoDIAN'].apply(lambda x: 1 if x == 'Documento Rechazado' else 0)

# Seleccionar características y objetivo para la regresión logística
X_rejection = new_data[['dia', 'documentoTipo']]
y_rejection = new_data['rechazado']

# Separar en entrenamiento y prueba
X_rej_train, X_rej_test, y_rej_train, y_rej_test = train_test_split(X_rejection, y_rejection, test_size=0.3, random_state=0)

# Entrenar el modelo de regresión logística
log_reg_rejection = LogisticRegression(max_iter=200)
log_reg_rejection.fit(X_rej_train, y_rej_train)

# Calcular el promedio diario de documentos por tipo
average_daily_documents_by_type = new_data.groupby('documentoTipo')['dia'].count() / len(new_data['dia'].unique())

# Generar proyecciones de rechazo por tipo de documento para los días 29 y 30 de octubre
projection_by_type = pd.DataFrame(columns=['documentoTipo', 'rechazo_prob', 'proj_rejections_29', 'proj_rejections_30'])
for doc_type in average_daily_documents_by_type.index:
    X_type = pd.DataFrame({'dia': [29, 30], 'documentoTipo': [doc_type, doc_type]})
    rejection_prob = log_reg_rejection.predict_proba(X_type)[:, 1]
    proj_rejections_29 = average_daily_documents_by_type[doc_type] * rejection_prob[0]
    proj_rejections_30 = average_daily_documents_by_type[doc_type] * rejection_prob[1]
    
    # Crear un DataFrame temporal y concatenarlo con `projection_by_type`
    temp_df = pd.DataFrame({
        'documentoTipo': [doc_type],
        'rechazo_prob': [rejection_prob[0]],
        'proj_rejections_29': [proj_rejections_29],
        'proj_rejections_30': [proj_rejections_30]
    })
    
    projection_by_type = pd.concat([projection_by_type, temp_df], ignore_index=True)


# Convertir documentoTipo a entero para visualización
doc_types = projection_by_type['documentoTipo'].astype(int)
proj_rej_29 = projection_by_type['proj_rejections_29']
proj_rej_30 = projection_by_type['proj_rejections_30']

# Gráfica para el 29 de octubre
plt.figure(figsize=(10, 6))
plt.bar(doc_types, proj_rej_29, color='steelblue')
plt.xlabel('Tipo de Documento')
plt.ylabel('Proyección de Documentos Rechazados')
plt.title('Proyección de Documentos Rechazados por Tipo para Octubre 29')
plt.ylim(0, max(proj_rej_29) * 1.5)
plt.xticks(doc_types, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()