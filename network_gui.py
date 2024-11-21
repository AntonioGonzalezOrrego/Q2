import tkinter as tk
from tkinter import messagebox
from genetic_network_design import NetworkOptimizationGA

class NetworkOptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimización de Diseño de Red - Algoritmo Genético")

        # Parámetros iniciales
        self.num_nodes = 5
        self.population_size = 10
        self.mutation_rate = 0.1
        self.generations = 1000

        self.create_widgets()

    def create_widgets(self):
        # Etiquetas y campos de entrada
        tk.Label(self.root, text="Número de nodos:").grid(row=0, column=0)
        self.num_nodes_entry = tk.Entry(self.root)
        self.num_nodes_entry.insert(0, str(self.num_nodes))
        self.num_nodes_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Tamaño de población:").grid(row=1, column=0)
        self.pop_size_entry = tk.Entry(self.root)
        self.pop_size_entry.insert(0, str(self.population_size))
        self.pop_size_entry.grid(row=1, column=1)

        tk.Label(self.root, text="Tasa de mutación:").grid(row=2, column=0)
        self.mutation_rate_entry = tk.Entry(self.root)
        self.mutation_rate_entry.insert(0, str(self.mutation_rate))
        self.mutation_rate_entry.grid(row=2, column=1)

        tk.Label(self.root, text="Número de generaciones:").grid(row=3, column=0)
        self.generations_entry = tk.Entry(self.root)
        self.generations_entry.insert(0, str(self.generations))
        self.generations_entry.grid(row=3, column=1)

        # Botón de ejecutar el algoritmo
        self.run_button = tk.Button(self.root, text="Ejecutar Algoritmo", command=self.run_algorithm)
        self.run_button.grid(row=4, column=0, columnspan=2)

        # Etiquetas de resultados
        self.result_label = tk.Label(self.root, text="Resultado:")
        self.result_label.grid(row=5, column=0, columnspan=2)

    def run_algorithm(self):
        # Obtener valores de los campos
        try:
            self.num_nodes = int(self.num_nodes_entry.get())
            self.population_size = int(self.pop_size_entry.get())
            self.mutation_rate = float(self.mutation_rate_entry.get())
            self.generations = int(self.generations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores válidos.")
            return

        # Crear la instancia del algoritmo genético
        ga = NetworkOptimizationGA(self.num_nodes, self.population_size, self.mutation_rate, self.generations)

        # Ejecutar el algoritmo genético
        best_individual, best_fitness = ga.run()

        # Mostrar el resultado
        self.result_label.config(text=f"Mejor configuración de red: {best_individual}\nCosto total: {best_fitness}")

# Ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkOptimizationApp(root)
    root.mainloop()
