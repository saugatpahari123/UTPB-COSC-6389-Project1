import random
import math
from collections import deque


class Candidate:
    """Represents an individual candidate in the population."""

    def __init__(self, chromosome, fitness=0.0):
        self.chromosome = chromosome
        self.fitness = fitness

    def calculate_fitness(self, fitness_function):
        """Calculates and updates the candidate's fitness using the provided fitness function."""
        self.fitness = fitness_function(self.chromosome)


def get_random_population(pop_size=20, gene_size=50):
    """Generates a random initial population."""
    population = [
        Candidate([random.randint(0, 100) for _ in range(gene_size)], random.uniform(0.0, 1.0))
        for _ in range(pop_size)
    ]
    for idx, candidate in enumerate(population):
        print(f"Candidate {idx + 1}: Chromosome = {candidate.chromosome[:5]}..., Fitness = {candidate.fitness:.4f}")
    return population


def hill_climb(candidate, fitness_function, max_iterations=1000):
    """Performs Hill Climbing on a given Candidate."""
    candidate.calculate_fitness(fitness_function)

    for _ in range(max_iterations):
        neighbor_chromosome = candidate.chromosome[:]
        index_to_modify = random.randint(0, len(neighbor_chromosome) - 1)
        neighbor_chromosome[index_to_modify] = random.randint(0, 100)

        neighbor = Candidate(neighbor_chromosome)
        neighbor.calculate_fitness(fitness_function)

        if neighbor.fitness > candidate.fitness:
            candidate = neighbor
    return candidate


def test_hill_climb():
    def example_fitness_function(chromosome):
        return sum(chromosome)

    initial_candidate = Candidate([random.randint(0, 100) for _ in range(50)])
    best_candidate = hill_climb(initial_candidate, example_fitness_function)

    print(f"Best Chromosome: {best_candidate.chromosome}")
    print(f"Best Fitness: {best_candidate.fitness}")


def simulated_annealing(candidate, fitness_function, initial_temperature=1000, cooling_rate=0.003,
                        min_temperature=1e-5):
    """Performs Simulated Annealing on a Candidate."""
    candidate.calculate_fitness(fitness_function)
    current_temperature = initial_temperature
    best_candidate = candidate

    while current_temperature > min_temperature:
        neighbor_chromosome = candidate.chromosome[:]
        index_to_modify = random.randint(0, len(neighbor_chromosome) - 1)
        neighbor_chromosome[index_to_modify] = random.randint(0, 100)

        neighbor = Candidate(neighbor_chromosome)
        neighbor.calculate_fitness(fitness_function)

        fitness_diff = neighbor.fitness - candidate.fitness
        if fitness_diff > 0 or random.random() < math.exp(fitness_diff / current_temperature):
            candidate = neighbor
            if neighbor.fitness > best_candidate.fitness:
                best_candidate = neighbor

        current_temperature *= (1 - cooling_rate)

    return best_candidate


def test_simulated_annealing():
    def example_fitness_function(chromosome):
        return sum(chromosome)

    initial_candidate = Candidate([random.randint(0, 100) for _ in range(50)])
    best_candidate = simulated_annealing(initial_candidate, example_fitness_function)

    print(f"Best Chromosome: {best_candidate.chromosome}")
    print(f"Best Fitness: {best_candidate.fitness}")


def tabu_search(initial_candidate, fitness_function, tabu_list_size=10, max_iterations=100, neighborhood_size=10):
    """Performs Tabu Search on a given Candidate."""
    initial_candidate.calculate_fitness(fitness_function)
    current_candidate = best_candidate = initial_candidate
    tabu_list = deque(maxlen=tabu_list_size)
    tabu_list.append(tuple(current_candidate.chromosome))

    for _ in range(max_iterations):
        neighborhood = []

        for _ in range(neighborhood_size):
            neighbor_chromosome = current_candidate.chromosome[:]
            index_to_modify = random.randint(0, len(neighbor_chromosome) - 1)
            neighbor_chromosome[index_to_modify] = random.randint(0, 100)

            neighbor = Candidate(neighbor_chromosome)
            neighbor.calculate_fitness(fitness_function)
            neighborhood.append(neighbor)

        best_neighbor = None
        for neighbor in neighborhood:
            if tuple(neighbor.chromosome) not in tabu_list or neighbor.fitness > best_candidate.fitness:
                if best_neighbor is None or neighbor.fitness > best_neighbor.fitness:
                    best_neighbor = neighbor

        if best_neighbor and best_neighbor.fitness > current_candidate.fitness:
            current_candidate = best_neighbor
            if best_neighbor.fitness > best_candidate.fitness:
                best_candidate = best_neighbor

            tabu_list.append(tuple(current_candidate.chromosome))

    return best_candidate


def test_tabu_search():
    def example_fitness_function(chromosome):
        return sum(chromosome)

    initial_candidate = Candidate([random.randint(0, 100) for _ in range(50)])
    best_candidate = tabu_search(initial_candidate, example_fitness_function)

    print(f"Best Chromosome: {best_candidate.chromosome}")
    print(f"Best Fitness: {best_candidate.fitness}")


def roulette_wheel_selection(generation):
    """Selects two parents using Roulette Wheel Selection."""
    total_fitness = sum(candidate.fitness for candidate in generation)

    def select_one():
        pick = random.uniform(0, total_fitness)
        current = 0
        for candidate in generation:
            current += candidate.fitness
            if current > pick:
                return candidate

    parent1 = select_one()
    parent2 = select_one()
    while parent2 == parent1:
        parent2 = select_one()
    return parent1, parent2


def rank_based_selection(generation):
    """Selects two parents using Rank-Based Selection."""
    ranked_generation = sorted(generation, key=lambda c: c.fitness)
    total_ranks = sum(range(1, len(ranked_generation) + 1))

    def select_one():
        pick = random.uniform(0, total_ranks)
        current = 0
        for i, candidate in enumerate(ranked_generation):
            current += (i + 1)
            if current > pick:
                return candidate

    parent1 = select_one()
    parent2 = select_one()
    return parent1, parent2


def n_point_crossover(parent1, parent2, n_points=2):
    """Performs N-point Crossover to create offspring from two parents."""
    length = len(parent1.chromosome)
    crossover_points = sorted(random.sample(range(1, length), n_points))
    offspring_chromosome = []
    swap = False
    prev_point = 0

    for point in crossover_points + [length]:
        offspring_chromosome.extend(
            parent2.chromosome[prev_point:point] if swap else parent1.chromosome[prev_point:point])
        swap = not swap
        prev_point = point

    return Candidate(offspring_chromosome)


def uniform_mutation(candidate, mutation_probability):
    """Applies Uniform Mutation to a Candidate."""
    offspring_chromosome = [
        random.randint(0, 100) if random.random() < mutation_probability else gene
        for gene in candidate.chromosome
    ]
    return Candidate(offspring_chromosome)


# Additional methods for mutation, selection, crossover, etc., would follow a similar structure.

# Test functions to check the behavior of each function individually.
if __name__ == "__main__":
    test_hill_climb()
    test_simulated_annealing()
    test_tabu_search()
