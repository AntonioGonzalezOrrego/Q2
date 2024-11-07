import pandas as pd
import matplotlib.pyplot as plt

# Cargar y preparar los datos
file_path = r'C:\Users\Admin\Documents\IA\Q2\BaseDatosFacturaElectronica.csv'
data = pd.read_csv(file_path)
data['documentoFecha'] = pd.to_datetime(data['documentoFecha'], errors='coerce')

# Filtrar solo los documentos rechazados
rechazados_data = data[data['documentoEstadoDIAN'] == 'Documento Rechazado']

# Agrupar por día para contar la cantidad de documentos rechazados
rechazados_data['dia'] = rechazados_data['documentoFecha'].dt.day  # Extraer el día del mes
rechazos_diarios = rechazados_data.groupby('dia').size()

# Crear el gráfico de análisis temporal de documentos rechazados por día
plt.figure(figsize=(10, 6))
rechazos_diarios.plot(kind='line', marker='o', color='red')
plt.title('Análisis Diario de Documentos Rechazados')
plt.xlabel('Día del Mes')
plt.ylabel('Cantidad de Documentos Rechazados')
plt.xticks(rechazos_diarios.index)  # Asegurar que todos los días del mes se muestran en el eje x
plt.grid(True)
plt.tight_layout()
plt.show()