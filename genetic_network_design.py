import random

class NetworkOptimizationGA:
    def __init__(self, num_nodes, population_size=10, mutation_rate=0.1, generations=1000):
        self.num_nodes = num_nodes
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        self.distances = [[random.randint(1, 100) for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Generar una configuración aleatoria de la red (conexiones)
    def generate_individual(self):
        return [random.sample(range(self.num_nodes), 2) for _ in range(self.num_nodes // 2)]  # Conexiones aleatorias

    # Función de evaluación: calculamos el costo total de la red (suma de distancias de las conexiones)
    def fitness(self, individual):
        cost = 0
        for connection in individual:
            cost += self.distances[connection[0]][connection[1]]
        return cost

    # Cruce entre dos individuos (configuraciones de red)
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    # Mutación: se altera una conexión aleatoria
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(individual) - 1)
            individual[index] = random.sample(range(self.num_nodes), 2)  # Cambia la conexión aleatoriamente
        return individual

    # Generar la población inicial
    def generate_population(self):
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    # Algoritmo genético
    def run(self):
        self.generate_population()
        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.generations):
            self.population.sort(key=lambda x: self.fitness(x))  # Ordenamos según la adaptación
            if self.fitness(self.population[0]) < best_fitness:
                best_individual = self.population[0]
                best_fitness = self.fitness(best_individual)

            # Selección: elegimos los mejores individuos
            selected = self.population[:self.population_size // 2]

            # Cruce y mutación para crear nueva población
            next_generation = selected
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            self.population = next_generation

        return best_individual, best_fitness
