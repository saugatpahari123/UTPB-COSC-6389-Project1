import math
import random
import tkinter as tk
from tkinter import Menu, Canvas, FALSE

# Configuration parameters
NUM_ITEMS = 100
FRAC_TARGET = 0.7
MIN_VALUE = 128
MAX_VALUE = 2048

SCREEN_PADDING = 25
ITEM_PADDING = 5
STROKE_WIDTH = 5

NUM_GENERATIONS = 1000
POP_SIZE = 50
ELITISM_COUNT = 2
MUTATION_RATE = 0.05
SLEEP_TIME = 100  # in milliseconds

# Helper function to generate a random RGB color
def random_rgb_color():
    red = random.randint(0x10, 0xff)
    green = random.randint(0x10, 0xff)
    blue = random.randint(0x10, 0xff)
    return f'#{red:02x}{green:02x}{blue:02x}'

# Class to represent an item with value and color
class Item:
    def __init__(self):
        self.value = random.randint(MIN_VALUE, MAX_VALUE)
        self.color = random_rgb_color()
        self.x, self.y, self.w, self.h = 0, 0, 0, 0

    def place(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def draw(self, canvas, active=False):
        color = self.color if active else ""
        canvas.create_rectangle(self.x, self.y, self.x + self.w, self.y + self.h, fill=color, outline=self.color, width=STROKE_WIDTH)
        canvas.create_text(self.x + self.w + 14, self.y + self.h / 2, text=f'{self.value}', anchor='w', font=('Arial', 12), fill='white')

# Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, items_list, target, pop_size, num_generations, mutation_rate, elitism_count):
        self.items_list = items_list
        self.target = target
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.population = []
        self.generation = 0
        self.best_genome = None
        self.running = False

    def gene_sum(self, genome):
        return sum(item.value for idx, item in enumerate(self.items_list) if genome[idx])

    def fitness(self, genome):
        total_value = self.gene_sum(genome)
        return 1 / (1 + abs(self.target - total_value))

    def generate_initial_population(self):
        self.population = [[random.random() < FRAC_TARGET for _ in range(len(self.items_list))] for _ in range(self.pop_size)]

    def select_parents(self, population, fitnesses, tournament_size=3):
        def tournament():
            competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
            return max(competitors, key=lambda x: x[1])[0]
        return tournament(), tournament()

    def crossover(self, parent1, parent2):
        return [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]

    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = not genome[i]
        return genome

    def evolve_population(self):
        fitnesses = [self.fitness(genome) for genome in self.population]
        sorted_population = sorted(zip(self.population, fitnesses), key=lambda x: x[1], reverse=True)
        new_population = [genome for genome, _ in sorted_population[:self.elitism_count]]

        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents([p for p, _ in sorted_population], [f for _, f in sorted_population])
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.best_genome = sorted_population[0][0]

    def run_step(self):
        if self.generation == 0:
            self.generate_initial_population()
        self.evolve_population()
        self.generation += 1
        return self.best_genome, self.generation

# UI Class for the Knapsack Problem
class KnapsackUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Knapsack Problem")
        self.option_add("*tearOff", FALSE)
        self.width, self.height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{self.width}x{self.height}+0+0")
        self.state("zoomed")

        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=self.width, height=self.height)
        self.items_list = []
        self.ga = None
        self.target = 0

        # Menu bar setup
        menu_bar = Menu(self)
        self.config(menu=menu_bar)
        menu_knapsack = Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(menu=menu_knapsack, label='Knapsack')
        menu_knapsack.add_command(label="Generate", command=self.generate_knapsack)
        menu_knapsack.add_command(label="Set Target", command=self.set_target)
        menu_knapsack.add_command(label="Run", command=self.start_ga)

    def get_random_item(self):
        item = Item()
        return item if item.value not in [i.value for i in self.items_list] else None

    def add_item(self):
        item = self.get_random_item()
        while not item:
            item = self.get_random_item()
        self.items_list.append(item)

    def generate_knapsack(self):
        self.items_list.clear()
        for _ in range(NUM_ITEMS):
            self.add_item()
        self.arrange_items()
        self.draw_items()

    def arrange_items(self):
        """Arrange items in the canvas based on their values."""
        w = self.width - SCREEN_PADDING
        h = self.height - SCREEN_PADDING
        num_rows = math.ceil(NUM_ITEMS / 6)
        row_w = w / 8 - ITEM_PADDING
        row_h = (h - 200) / num_rows
        item_max = max(item.value for item in self.items_list)

        for x in range(6):
            for y in range(num_rows):
                idx = x * num_rows + y
                if idx >= len(self.items_list):
                    break
                item = self.items_list[idx]
                item.place(SCREEN_PADDING + x * row_w + x * ITEM_PADDING, SCREEN_PADDING + y * row_h + y * ITEM_PADDING, row_w / 2, max(item.value / item_max * row_h, 1))

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw_items(self):
        self.clear_canvas()
        for item in self.items_list:
            item.draw(self.canvas)

    def draw_target(self):
        x, y = (self.width - SCREEN_PADDING) * 7 / 8, SCREEN_PADDING
        w, h = (self.width - SCREEN_PADDING) / 8 - SCREEN_PADDING, self.height / 2 - SCREEN_PADDING
        self.canvas.create_rectangle(x, y, x + w, y + h, fill='yellow')
        self.canvas.create_text(x + w // 2, y + h + SCREEN_PADDING, text=f'Target: {int(self.target)}', font=('Arial', 18, 'bold'), fill='green')

    def draw_sum(self, item_sum):
        x, y = (self.width - SCREEN_PADDING) * 6 / 8, SCREEN_PADDING
        w, h = (self.width - SCREEN_PADDING) / 8 - SCREEN_PADDING, self.height / 2 - SCREEN_PADDING
        h = h * (item_sum / self.target) if self.target != 0 else 0
        self.canvas.create_rectangle(x, y, x + w, y + h, fill='orange')
        self.canvas.create_text(x + w // 2, y + h + SCREEN_PADDING, text=f'Sum: {int(item_sum)}', font=('Arial', 18, 'bold'), fill='yellow')

    def draw_genome(self, genome, generation):
        for idx, item in enumerate(self.items_list):
            item.draw(self.canvas, active=genome[idx])
        x, y = (self.width - SCREEN_PADDING) * 6 / 8, SCREEN_PADDING
        w, h = (self.width - SCREEN_PADDING) / 8 - SCREEN_PADDING, self.height / 4 * 3
        self.canvas.create_text(x + w, y + h + SCREEN_PADDING * 2, text=f'Generation {generation}', font=('Arial', 18, 'bold'), fill='red')

    def get_item_sum(self, genome):
        return sum(item.value for idx, item in enumerate(self.items_list) if genome[idx])

    def set_target(self):
        target_set = random.sample(self.items_list, int(NUM_ITEMS * FRAC_TARGET))
        self.target = sum(item.value for item in target_set)
        self.draw_items()
        self.draw_target()

    def start_ga(self):
        if self.target == 0:
            self.set_target()
        self.ga = GeneticAlgorithm(self.items_list, self.target, POP_SIZE, NUM_GENERATIONS, MUTATION_RATE, ELITISM_COUNT)
        self.run_ga_step()

    def run_ga_step(self):
        if self.ga.generation == 0:
            self.ga.generate_initial_population()
        self.ga.evolve_population()
        best_genome, generation = self.ga.best_genome, self.ga.generation
        item_sum = self.get_item_sum(best_genome)

        # Draw the updated state
        self.draw_items()
        self.draw_genome(best_genome, generation)
        self.draw_target()
        self.draw_sum(item_sum)

        # Print current generation information
        print(f'Generation {generation}, Sum: {item_sum}, Fitness: {self.ga.fitness(best_genome)}')

        # Check if target is met or max generations reached
        if item_sum == self.target:
            print('Exact solution found!')
        elif generation < NUM_GENERATIONS:
            self.after(SLEEP_TIME, self.run_ga_step)
        else:
            print('Algorithm finished.')

# Run the application
if __name__ == '__main__':
    ui = KnapsackUI()
    ui.mainloop()
