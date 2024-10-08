import tkinter as tk
from tkinter import messagebox
import random

# Parámetros generales
NUM_EMPLOYEES = 5
NUM_TASKS = 5
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1

# Algoritmo genético
class GeneticAlgorithm:
    def __init__(self, time_matrix):
        self.time_matrix = time_matrix

    def run(self):
        population = self.initialize_population(POPULATION_SIZE)
        best_solution = None
        best_fitness = float('inf')

        for generation in range(GENERATIONS):
            # Evaluar la población
            fitness_population = [(individual, self.calculate_fitness(individual)) for individual in population]
            fitness_population.sort(key=lambda x: x[1])

            # Actualizar la mejor solución
            if fitness_population[0][1] < best_fitness:
                best_solution, best_fitness = fitness_population[0]

            # Generar nueva población
            population = self.generate_new_population(fitness_population)

        return best_solution, best_fitness

    def initialize_population(self, size):
        population = []
        for _ in range(size):
            assignment = list(range(NUM_TASKS))
            random.shuffle(assignment)
            population.append(assignment)
        return population

    def calculate_fitness(self, assignment):
        total_time = 0
        for i in range(NUM_TASKS):
            total_time += self.time_matrix[i][assignment[i]]
        return total_time

    def generate_new_population(self, fitness_population):
        new_population = []
        # Elitismo: guardar el mejor individuo
        new_population.append(fitness_population[0][0])

        # Crear nueva población
        while len(new_population) < POPULATION_SIZE:
            parent1 = self.select_parent(fitness_population)
            parent2 = self.select_parent(fitness_population)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        return new_population

    def select_parent(self, fitness_population):
        tournament_size = 5
        tournament = random.sample(fitness_population, tournament_size)
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, NUM_TASKS - 2)
        child = parent1[:crossover_point] + [task for task in parent2 if task not in parent1[:crossover_point]]
        return child

    def mutate(self, individual):
        if random.random() < MUTATION_RATE:
            i, j = random.sample(range(NUM_TASKS), 2)
            individual[i], individual[j] = individual[j], individual[i]

# Interfaz gráfica
class TaskAssignmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Asignar tareas para el algoritmo genético")

        # Crear la cuadrícula de entrada de tiempos
        self.time_matrix = [[tk.StringVar() for _ in range(NUM_TASKS)] for _ in range(NUM_EMPLOYEES)]
        for i in range(NUM_EMPLOYEES):
            for j in range(NUM_TASKS):
                entry = tk.Entry(root, textvariable=self.time_matrix[i][j], width=5)
                entry.grid(row=i, column=j, padx=5, pady=5)

        # Botón de ejecución
        self.run_button = tk.Button(root, text="Iniciar", command=self.run_algorithm)
        self.run_button.grid(row=NUM_EMPLOYEES, column=0, columnspan=NUM_TASKS, pady=10)

        # Etiqueta de resultados
        self.result_label = tk.Label(root, text="Solución:")
        self.result_label.grid(row=NUM_EMPLOYEES + 1, column=0, columnspan=NUM_TASKS, pady=10)

    def run_algorithm(self):
        # Cargar los tiempos de la cuadrícula
        time_matrix = []
        for i in range(NUM_EMPLOYEES):
            row = []
            for j in range(NUM_TASKS):
                try:
                    row.append(int(self.time_matrix[i][j].get()))
                except ValueError:
                    messagebox.showerror("Error", "Por favor ingresar números validos")
                    return
            time_matrix.append(row)

        # Ejecutar el algoritmo genético
        ga = GeneticAlgorithm(time_matrix)
        best_solution, best_fitness = ga.run()

        # Mostrar la solución
        solution_text = f"Mejor solución posible (Tiempo: {best_fitness}):\n"
        for i in range(NUM_EMPLOYEES):
            solution_text += f"El empleado {i + 1} -> debería realizar la tarea {best_solution[i] + 1}\n"

        self.result_label.config(text=solution_text)

# Crear la ventana principal
if __name__ == "__main__":
    root = tk.Tk()
    app = TaskAssignmentApp(root)
    root.mainloop()
