import random
import math
import tkinter as tk
from tkinter import Menu, Canvas, FALSE
import threading

# Configuration for the subset sum problem
NUM_ITEMS = 20
MIN_VALUE = 1
MAX_VALUE = 50
TARGET_VALUE = random.randint(50, 200)

# Backtracking function for Subset Sum Problem
def subset_sum(items, target, partial=[]):
    """Recursively finds a subset that sums to the target."""
    current_sum = sum(partial)

    # Check if we have a solution or exceeded the target
    if current_sum == target:
        return partial
    if current_sum > target:
        return None

    # Try adding each item in the list to the partial solution
    for i in range(len(items)):
        remaining = items[i + 1:]
        result = subset_sum(remaining, target, partial + [items[i]])
        if result is not None:
            return result
    return None

# Particle class for PSO
class Particle:
    def __init__(self, num_items):
        # Position (binary) indicating subset selection
        self.position = [random.choice([0, 1]) for _ in range(num_items)]
        # Velocity for each dimension
        self.velocity = [random.uniform(-1, 1) for _ in range(num_items)]
        self.best_position = self.position[:]
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        """Updates particle velocity based on cognitive and social components."""
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self):
        """Updates particle position using sigmoid function."""
        for i in range(len(self.position)):
            if random.random() < 1 / (1 + math.exp(-self.velocity[i])):
                self.position[i] = 1
            else:
                self.position[i] = 0

# PSO Solver class for Subset Sum Problem
class PSOSolver:
    def __init__(self, items, target, num_particles=30, max_iterations=100):
        self.items = items
        self.target = target
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.particles = [Particle(len(items)) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')

    def fitness(self, position):
        """Evaluates the fitness of a particle based on the subset sum distance from the target."""
        subset_sum = sum(self.items[i] for i in range(len(position)) if position[i] == 1)
        return abs(self.target - subset_sum)

    def solve(self):
        """Performs PSO to find a subset that sums close to the target."""
        for _ in range(self.max_iterations):
            for particle in self.particles:
                current_value = self.fitness(particle.position)

                # Update particle's personal best
                if current_value < particle.best_value:
                    particle.best_value = current_value
                    particle.best_position = particle.position[:]

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = particle.position[:]

            # Update particles' velocity and position
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w=0.5, c1=1.5, c2=1.5)
                particle.update_position()

        # Return solution if exact match found
        return self.global_best_position if self.global_best_value == 0 else None

# Main UI Class for Subset Sum Problem
class SubsetSumUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Subset Sum Problem")
        self.option_add("*tearOff", FALSE)
        self.width, self.height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{self.width}x{self.height}+0+0")
        self.state("zoomed")

        # Setup canvas and menu
        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=self.width, height=self.height)
        self.create_menu()

        self.items_list = []
        self.target = TARGET_VALUE
        self.solution = None

    def create_menu(self):
        """Creates the main menu bar with options for generating and solving the subset sum problem."""
        menu_bar = Menu(self)
        self.config(menu=menu_bar)
        subset_sum_menu = Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(menu=subset_sum_menu, label='Subset Sum')

        subset_sum_menu.add_command(label="Generate Set", command=self.generate_set)
        subset_sum_menu.add_command(label="Solve with Backtracking", command=self.start_backtracking_solver)
        subset_sum_menu.add_command(label="Solve with PSO", command=self.start_pso_solver)

    def generate_set(self):
        """Generates a random set of items for the subset sum problem."""
        self.items_list = [random.randint(MIN_VALUE, MAX_VALUE) for _ in range(NUM_ITEMS)]
        self.clear_canvas()
        self.draw_target()
        self.draw_items()

    def clear_canvas(self):
        """Clears the canvas for new drawings."""
        self.canvas.delete("all")

    def draw_items(self):
        """Draws the items on the canvas."""
        x_start, y_start = 100, 150
        for i, value in enumerate(self.items_list):
            self.canvas.create_rectangle(x_start, y_start + i * 30, x_start + 100, y_start + (i + 1) * 30, fill='lightblue')
            self.canvas.create_text(x_start + 50, y_start + i * 30 + 15, text=str(value), font=('Arial', 14))

    def draw_target(self):
        """Displays the target sum on the canvas."""
        self.canvas.create_text(150, 100, text=f'Target: {self.target}', font=('Arial', 18), fill='darkorange')

    def start_backtracking_solver(self):
        """Starts the backtracking solver in a separate thread."""
        if not self.items_list:
            self.generate_set()
        threading.Thread(target=self.run_backtracking_solver).start()

    def run_backtracking_solver(self):
        """Runs the backtracking solver and displays the solution."""
        self.solution = subset_sum(self.items_list, self.target)
        self.after(0, self.draw_solution)

    def start_pso_solver(self):
        """Starts the PSO solver in a separate thread."""
        if not self.items_list:
            self.generate_set()
        threading.Thread(target=self.run_pso_solver).start()

    def run_pso_solver(self):
        """Runs the PSO solver and displays the solution."""
        pso_solver = PSOSolver(self.items_list, self.target)
        pso_solution = pso_solver.solve()
        self.solution = [self.items_list[i] for i in range(len(self.items_list)) if pso_solution and pso_solution[i] == 1]
        self.after(0, self.draw_solution)

    def draw_solution(self):
        """Displays the solution on the canvas."""
        self.clear_canvas()
        self.draw_target()
        self.draw_items()

        # Check if a solution was found and display appropriately
        if not self.solution:
            self.canvas.create_text(400, 100, text='No Solution Found', font=('Arial', 18), fill='red')
        else:
            x_start, y_start = 250, 150
            self.canvas.create_text(300, 100, text='Solution Found:', font=('Arial', 18), fill='green')
            for i, value in enumerate(self.solution):
                self.canvas.create_rectangle(x_start, y_start + i * 30, x_start + 100, y_start + (i + 1) * 30, fill='lightgreen')
                self.canvas.create_text(x_start + 50, y_start + i * 30 + 15, text=str(value), font=('Arial', 14))

# Run the application
if __name__ == '__main__':
    ui = SubsetSumUI()
    ui.mainloop()
