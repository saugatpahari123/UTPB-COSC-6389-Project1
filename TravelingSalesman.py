import math
import random
import tkinter as tk
from tkinter import Menu, Canvas, FALSE

# Configuration parameters
NUM_CITIES = 25
CITY_SCALE = 5
ROAD_WIDTH = 2
PADDING = 50


class Node:
    """Represents a city node with an (x, y) position and an index identifier."""

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def draw(self, canvas, color='yellow'):
        """Draws the city node on the canvas."""
        canvas.create_oval(
            self.x - CITY_SCALE * 2, self.y - CITY_SCALE * 2,
            self.x + CITY_SCALE * 2, self.y + CITY_SCALE * 2,
            fill=color, outline='black'
        )
        canvas.create_text(
            self.x, self.y - CITY_SCALE * 3,
            text=str(self.index),
            font=('Arial', 12),
            fill='blue'
        )


class Edge:
    """Represents an edge between two cities."""

    def __init__(self, a, b):
        self.city_a = a
        self.city_b = b
        self.length = math.hypot(a.x - b.x, a.y - b.y)

    def draw(self, canvas, color='grey', style=None):
        """Draws the edge on the canvas."""
        kwargs = {'fill': color, 'width': ROAD_WIDTH}
        if style:
            kwargs['dash'] = style
        canvas.create_line(
            self.city_a.x, self.city_a.y,
            self.city_b.x, self.city_b.y,
            **kwargs
        )


class TSPSolver:
    """Solver for the Traveling Salesman Problem using Simulated Annealing."""

    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self.calculate_distance_matrix()
        self.current_solution = list(range(self.num_cities))
        random.shuffle(self.current_solution)
        self.best_solution = self.current_solution[:]
        self.best_distance = self.calculate_total_distance(self.best_solution)
        self.temperature = 10000
        self.cooling_rate = 0.995

    def calculate_distance_matrix(self):
        """Calculates the distance matrix for all city pairs."""
        matrix = [[0] * self.num_cities for _ in range(self.num_cities)]
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = math.hypot(
                    self.cities[i].x - self.cities[j].x,
                    self.cities[i].y - self.cities[j].y
                )
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def calculate_total_distance(self, solution):
        """Calculates the total distance of the given solution."""
        return sum(
            self.distance_matrix[solution[i]][solution[(i + 1) % len(solution)]]
            for i in range(len(solution))
        )

    def swap_cities(self, solution):
        """Creates a new solution by swapping two cities."""
        new_solution = solution[:]
        i, j = random.sample(range(self.num_cities), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def anneal(self):
        """Performs one iteration of the annealing process."""
        new_solution = self.swap_cities(self.current_solution)
        current_distance = self.calculate_total_distance(self.current_solution)
        new_distance = self.calculate_total_distance(new_solution)
        acceptance_prob = self.acceptance_probability(current_distance, new_distance, self.temperature)

        if acceptance_prob > random.random():
            self.current_solution = new_solution
            current_distance = new_distance
            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_solution = self.current_solution[:]

        self.temperature *= self.cooling_rate

    def acceptance_probability(self, current_distance, new_distance, temperature):
        """Calculates the probability of accepting a worse solution."""
        if new_distance < current_distance:
            return 1.0
        return math.exp((current_distance - new_distance) / temperature)


class TSPUI(tk.Tk):
    """User interface for visualizing the TSP solution."""

    def __init__(self):
        super().__init__()
        self.title("Traveling Salesman Problem")
        self.option_add("*tearOff", FALSE)
        self.width, self.height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{self.width}x{self.height}+0+0")
        self.state("zoomed")

        # Canvas and setup
        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=self.width, height=self.height)
        self.w = self.width - PADDING * 2
        self.h = self.height - PADDING * 2

        # City list and solver
        self.cities_list = []
        self.tsp_solver = None
        self.is_running = False

        # Menu bar setup
        self.create_menu()

    def create_menu(self):
        """Creates the main menu for generating cities and running the solver."""
        menu_bar = Menu(self)
        self.config(menu=menu_bar)
        tsp_menu = Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(menu=tsp_menu, label='Salesman')

        tsp_menu.add_command(label="Generate", command=self.generate)
        tsp_menu.add_command(label="Run", command=self.start_solver)

    def generate(self):
        """Generates random cities on the canvas."""
        self.clear_canvas()
        self.cities_list.clear()
        for i in range(NUM_CITIES):
            self.add_city(i)
        self.draw_cities()

    def add_city(self, index):
        """Adds a city with a random location to the cities list."""
        x = random.randint(PADDING, self.w)
        y = random.randint(PADDING, self.h)
        node = Node(x, y, index)
        self.cities_list.append(node)

    def draw_cities(self):
        """Draws all cities on the canvas."""
        for city in self.cities_list:
            city.draw(self.canvas)

    def clear_canvas(self):
        """Clears the canvas."""
        self.canvas.delete("all")

    def start_solver(self):
        """Starts the TSP solver using Simulated Annealing."""
        if not self.cities_list:
            self.generate()
        self.tsp_solver = TSPSolver(self.cities_list)
        self.is_running = True
        self.run_solver()

    def run_solver(self):
        """Runs the solver and updates the visualization in real-time."""
        if self.is_running and self.tsp_solver.temperature > 1:
            self.tsp_solver.anneal()
            self.clear_canvas()
            self.draw_solution(self.tsp_solver.current_solution)
            self.canvas.update()
            self.after(1, self.run_solver)
        else:
            self.is_running = False
            print(f"Best distance found: {self.tsp_solver.best_distance}")
            self.display_best_distance()

    def display_best_distance(self):
        """Displays the best distance found on the canvas."""
        self.canvas.create_text(
            PADDING, PADDING,
            text=f"Best Distance Found: {int(self.tsp_solver.best_distance)}",
            font=('Arial', 20, 'bold'),
            fill='green',
            anchor='nw'
        )

    def draw_solution(self, solution):
        """Draws the current solution path on the canvas."""
        for i in range(len(solution)):
            city_a = self.cities_list[solution[i]]
            city_b = self.cities_list[solution[(i + 1) % len(solution)]]
            edge = Edge(city_a, city_b)
            edge.draw(self.canvas, color='red')

        # Draw the cities and the current distance
        for city in self.cities_list:
            city.draw(self.canvas, color='blue')
        self.canvas.create_text(
            PADDING, PADDING // 2,
            text=f"Distance: {int(self.tsp_solver.best_distance)}",
            font=('Arial', 20, 'bold'),
            fill='green',
            anchor='nw'
        )


if __name__ == '__main__':
    TSPUI().mainloop()
