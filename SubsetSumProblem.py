import math
import random
import tkinter as tk
from tkinter import *
import time

# Problem parameters
NUM_ITEMS = 50
MIN_VALUE = 10
MAX_VALUE = 100
TARGET_SUM = 1000
MAX_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
SLEEP_TIME = 0.2


class Item:
    """Represents an item with a random value."""

    def __init__(self):
        self.value = random.randint(MIN_VALUE, MAX_VALUE)


class SubsetSumApp(tk.Tk):
    """Main application for solving the Subset Sum problem."""

    def __init__(self):
        super().__init__()
        self.title("Subset Sum Problem Solver")
        self.geometry("1200x700")
        self.configure(bg="#282c34")

        # Solver state
        self.items = [Item() for _ in range(NUM_ITEMS)]
        self.target_sum = TARGET_SUM
        self.running = False
        self.selected_solver = tk.StringVar(value="GA")
        self.progress_data = []

        # Layout setup
        self.create_ui()

    def create_ui(self):
        """Create the user interface components."""
        # Title label
        title_label = Label(
            self,
            text="Subset Sum Problem Solver",
            font=("Arial", 18, "bold"),
            bg="#61afef",
            fg="#ffffff",
            pady=10
        )
        title_label.pack(fill=tk.X)

        # Solver selection menu
        solver_frame = tk.Frame(self, bg="#282c34")
        solver_frame.pack(pady=10)
        tk.Label(solver_frame, text="Select Solver:", bg="#282c34", fg="#ffffff", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(solver_frame, text="Genetic Algorithm", variable=self.selected_solver, value="GA", bg="#282c34", fg="#ffffff").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(solver_frame, text="Particle Swarm Optimization", variable=self.selected_solver, value="PSO", bg="#282c34", fg="#ffffff").pack(side=tk.LEFT, padx=5)

        # Items display
        self.items_label = Label(self, text="", font=("Arial", 12), bg="#61afef", fg="#282c34", wraplength=800, justify=tk.LEFT)
        self.items_label.pack(pady=10)
        self.update_items_display()

        # Control buttons
        button_frame = tk.Frame(self, bg="#282c34")
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Start", command=self.start_solver, bg="#98c379", fg="#282c34").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Stop", command=self.stop_solver, bg="#e06c75", fg="#282c34").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Report", command=self.show_report, bg="#56b6c2", fg="#282c34").pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = Label(self, text="Status: Waiting...", font=("Arial", 12), bg="#282c34", fg="#ffffff")
        self.status_label.pack(pady=10)

    def update_items_display(self):
        """Update the items display in the UI."""
        items_text = "\n".join(
            ", ".join(f"{item.value}" for item in self.items[i:i + 10])
            for i in range(0, len(self.items), 10)
        )
        self.items_label.config(text=f"Items:\n{items_text}")

    def start_solver(self):
        """Start the selected optimization solver."""
        self.running = True
        self.progress_data.clear()
        solver_type = self.selected_solver.get()

        if solver_type == "GA":
            GeneticAlgorithm(self.items, self.target_sum, self).run()
        elif solver_type == "PSO":
            ParticleSwarmOptimization(self.items, self.target_sum, self).run()

    def stop_solver(self):
        """Stop the solver."""
        self.running = False
        self.status_label.config(text="Status: Solver stopped.")

    def show_report(self):
        """Show the optimization progress report."""
        report_window = tk.Toplevel(self)
        report_window.title("Progress Report")
        report_text = tk.Text(report_window, wrap="word", font=("Arial", 10))
        report_text.pack(fill=tk.BOTH, expand=True)
        report_text.insert(tk.END, "Optimization Progress:\n\n")

        for generation, fitness in self.progress_data:
            report_text.insert(tk.END, f"Generation {generation}: Fitness = {fitness:.2f}\n")


class GeneticAlgorithm:
    """Genetic Algorithm for solving the Subset Sum problem."""

    def __init__(self, items, target, ui):
        self.items = items
        self.target = target
        self.ui = ui
        self.population = self.initialize_population()

    def initialize_population(self):
        """Generate an initial random population."""
        return [[random.choice([True, False]) for _ in range(len(self.items))] for _ in range(POPULATION_SIZE)]

    def fitness(self, genome):
        """Calculate the fitness of a genome."""
        total = sum(item.value for item, selected in zip(self.items, genome) if selected)
        return abs(total - self.target) if total <= self.target else float("inf")

    def select_parents(self):
        """Select two parents using tournament selection."""
        tournament_size = 5
        return min(random.sample(self.population, tournament_size), key=self.fitness), \
               min(random.sample(self.population, tournament_size), key=self.fitness)

    def crossover(self, parent1, parent2):
        """Perform single-point crossover."""
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]

    def mutate(self, genome):
        """Mutate a genome with a small probability."""
        for i in range(len(genome)):
            if random.random() < MUTATION_RATE:
                genome[i] = not genome[i]
        return genome

    def run(self):
        """Run the genetic algorithm."""
        for generation in range(MAX_GENERATIONS):
            if not self.ui.running:
                break

            # Evaluate fitness and track progress
            self.population.sort(key=self.fitness)
            best_genome = self.population[0]
            best_fitness = self.fitness(best_genome)
            self.ui.progress_data.append((generation, best_fitness))

            # Update UI
            selected_values = [item.value for item, selected in zip(self.items, best_genome) if selected]
            self.ui.status_label.config(text=f"Generation {generation}: Best Sum = {sum(selected_values)}")
            self.ui.update()

            if best_fitness == 0:  # Exact match found
                break

            # Create the next generation
            new_population = self.population[:2]  # Elitism
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child))

            self.population = new_population
            time.sleep(SLEEP_TIME)


class ParticleSwarmOptimization:
    """Particle Swarm Optimization for solving the Subset Sum problem."""

    def __init__(self, items, target, ui):
        self.items = items
        self.target = target
        self.ui = ui
        self.num_particles = POPULATION_SIZE
        self.particles = self.initialize_particles()
        self.velocities = [[0.0] * len(self.items) for _ in range(self.num_particles)]
        self.personal_best_positions = self.particles[:]
        self.personal_best_fitness = [self.fitness(p) for p in self.particles]
        self.global_best_position = min(self.particles, key=self.fitness)
        self.global_best_fitness = self.fitness(self.global_best_position)

    def initialize_particles(self):
        """Generate initial random particles."""
        return [[random.choice([True, False]) for _ in range(len(self.items))] for _ in range(self.num_particles)]

    def fitness(self, particle):
        """Calculate fitness of a particle."""
        total = sum(item.value for item, included in zip(self.items, particle) if included)
        if total > self.target:
            return (total - self.target) ** 2  # Penalize overshooting
        else:
            return abs(self.target - total)

    def update_velocity_and_position(self, particle_idx, inertia_weight):
        """Update the velocity and position of a particle."""
        cognitive_weight = 1.5
        social_weight = 1.5

        for i in range(len(self.particles[particle_idx])):
            r1 = random.random()
            r2 = random.random()

            # Update velocity
            self.velocities[particle_idx][i] = (
                inertia_weight * self.velocities[particle_idx][i]
                + cognitive_weight * r1 * (self.personal_best_positions[particle_idx][i] - self.particles[particle_idx][i])
                + social_weight * r2 * (self.global_best_position[i] - self.particles[particle_idx][i])
            )

            # Sigmoid function for binary decisions
            sigmoid = 1 / (1 + math.exp(-self.velocities[particle_idx][i]))
            self.particles[particle_idx][i] = random.random() < sigmoid

    def run(self):
        """Run the PSO algorithm."""
        max_stagnation = 50
        stagnation_counter = 0
        last_best_fitness = float("inf")

        for generation in range(MAX_GENERATIONS):
            if not self.ui.running:
                break

            inertia_weight = 0.9 - (generation / MAX_GENERATIONS) * (0.9 - 0.4)

            for idx in range(self.num_particles):
                self.update_velocity_and_position(idx, inertia_weight)

                # Update personal best
                current_fitness = self.fitness(self.particles[idx])
                if current_fitness < self.personal_best_fitness[idx]:
                    self.personal_best_positions[idx] = self.particles[idx][:]
                    self.personal_best_fitness[idx] = current_fitness

                # Update global best
                if current_fitness < self.global_best_fitness:
                    self.global_best_position = self.particles[idx][:]
                    self.global_best_fitness = current_fitness

            # Track progress
            if self.global_best_fitness != last_best_fitness:
                stagnation_counter = 0
                self.ui.progress_data.append((generation, self.global_best_fitness))
                last_best_fitness = self.global_best_fitness
            else:
                stagnation_counter += 1

            # Update UI
            subset = [item.value for item, included in zip(self.items, self.global_best_position) if included]
            subset_sum = sum(subset)
            self.ui.status_label.config(
                text=f"Generation: {generation}\nSubset: {subset}\nSum: {subset_sum}\nFitness: {self.global_best_fitness:.2f}",
                font=("Arial", 12),
                bg="#61afef",
                fg="#282c34"
            )
            self.ui.update()

            # Stop if stagnation or exact match
            if stagnation_counter >= max_stagnation or self.global_best_fitness == 0:
                print("Stopping due to stagnation or exact solution.")
                break

            time.sleep(SLEEP_TIME)



if __name__ == "__main__":
    app = SubsetSumApp()
    app.mainloop()
