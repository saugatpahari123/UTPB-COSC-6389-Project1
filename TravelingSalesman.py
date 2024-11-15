import math
import random
import tkinter as tk
from tkinter import *

num_cities = 50
city_scale = 5
padding = 100

# Parameters for ACO
alpha = 1.0  # Influence of pheromone
beta = 5.0  # Influence of heuristic value (visibility)
evaporation = 0.5  # Pheromone evaporation rate
ant_count = 20  # Number of ants
max_iterations = 100  # Maximum number of iterations for ACO
stagnation_limit = (
    20  # Number of iterations with no improvement to consider convergence
)

# Parameters for GA
population_size = 100
mutation_rate = 0.02
crossover_rate = 0.8
ga_iterations = 5000


class Node:
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index  # Unique identifier for the city

    def draw(self, canvas, color="black"):
        canvas.create_oval(
            self.x - city_scale,
            self.y - city_scale,
            self.x + city_scale,
            self.y + city_scale,
            fill=color,
            tags="cities",
        )


class Edge:
    def __init__(self, a, b):
        self.city_a = a
        self.city_b = b
        self.length = math.hypot(a.x - b.x, a.y - b.y)

    def draw(self, canvas, color="black", dash=(2, 4)):
        canvas.create_line(
            self.city_a.x,
            self.city_a.y,
            self.city_b.x,
            self.city_b.y,
            fill=color,
            width=1,
            dash=dash,
            tags="edges",
        )


class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Traveling Salesman")
        self.option_add("*tearOff", FALSE)
        width, height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (width, height))
        self.state("zoomed")

        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=width, height=height)
        w = width - padding
        h = height - padding * 2

        self.cities_list = []
        self.edges_list = []
        self.tour = []
        self.best_distance = None
        self.iteration = 0
        self.stagnant_iterations = 0  # For convergence check
        self.optimizing = False  # Flag to control optimization loop
        self.algorithm = "2-opt"  # Default algorithm
        self.pheromone = {}  # Pheromone levels for ACO
        self.population = []  # Population for GA
        self.status_label = tk.Label(self, text="Distance: 0.00")
        self.status_label.place(x=10, y=10)

        # Algorithm selection
        self.algorithm_var = tk.StringVar(value="2-opt")
        self.radio_2opt = tk.Radiobutton(
            self, text="2-opt", variable=self.algorithm_var, value="2-opt"
        )
        self.radio_aco = tk.Radiobutton(
            self, text="ACO", variable=self.algorithm_var, value="ACO"
        )
        self.radio_ga = tk.Radiobutton(
            self, text="GA", variable=self.algorithm_var, value="GA"
        )
        self.radio_2opt.place(x=10, y=40)
        self.radio_aco.place(x=70, y=40)
        self.radio_ga.place(x=130, y=40)

        # Solve button
        self.solve_button = tk.Button(
            self, text="Solve", command=self.start_optimization
        )
        self.solve_button.place(x=200, y=38)

        def generate_cities():
            self.cities_list.clear()
            self.edges_list.clear()
            self.tour.clear()
            self.best_distance = None
            self.iteration = 0
            self.stagnant_iterations = 0
            self.optimizing = False
            self.algorithm = self.algorithm_var.get()
            self.pheromone.clear()
            self.population.clear()
            # Generate cities
            for idx in range(num_cities):
                x = random.randint(padding, w)
                y = random.randint(padding, h)
                node = Node(x, y, idx)
                self.cities_list.append(node)
            # Generate all possible edges
            N = len(self.cities_list)
            for i in range(N):
                for j in range(i + 1, N):
                    edge = Edge(self.cities_list[i], self.cities_list[j])
                    self.edges_list.append(edge)
            draw_cities()
            # Do not initialize the tour or draw it here

        def draw_cities():
            self.canvas.delete("all")
            # Draw all edges as dotted black lines
            for edge in self.edges_list:
                edge.draw(self.canvas, color="black", dash=(2, 4))
            # Draw cities
            for n in self.cities_list:
                n.draw(self.canvas)
            # Do not draw the tour here

        # Menu setup
        menu_bar = Menu(self)
        self["menu"] = menu_bar

        # Generate menu
        menu_generate = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_generate, label="Generate", underline=0)
        menu_generate.add_command(
            label="Generate Cities", command=generate_cities, underline=0
        )

        # Optimize menu (Pause and Reset)
        menu_optimize = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_optimize, label="Optimize", underline=0)
        menu_optimize.add_command(
            label="Pause Optimization", command=self.pause_optimization, underline=0
        )
        menu_optimize.add_command(
            label="Reset Tour", command=self.reset_tour, underline=0
        )

        self.mainloop()

    def initialize_tour(self):
        self.tour = list(range(len(self.cities_list)))  # Initial tour in order
        self.best_distance = self.calculate_tour_distance(self.tour)
        self.iteration = 0
        self.stagnant_iterations = 0

    def calculate_tour_distance(self, tour):
        distance = 0
        N = len(tour)
        for i in range(N):
            a = self.cities_list[tour[i]]
            b = self.cities_list[tour[(i + 1) % N]]
            dx = a.x - b.x
            dy = a.y - b.y
            distance += math.hypot(dx, dy)
        return distance

    def start_optimization(self):
        if not self.optimizing:
            self.algorithm = self.algorithm_var.get()
            if not self.tour:
                if self.algorithm == "GA":
                    self.initialize_population()
                else:
                    self.initialize_tour()
            self.optimizing = True
            if self.algorithm == "2-opt":
                self.draw_current_tour()
                self.two_opt_iteration()
            elif self.algorithm == "ACO":
                self.initialize_pheromone()
                self.draw_current_tour()
                self.aco_iteration()
            elif self.algorithm == "GA":
                self.draw_current_tour()
                self.ga_iteration()

    def pause_optimization(self):
        self.optimizing = False

    def reset_tour(self):
        if self.algorithm == "GA":
            self.initialize_population()
        else:
            self.initialize_tour()
        self.draw_current_tour()

    def two_opt_iteration(self):
        if not self.optimizing or self.algorithm != "2-opt":
            return
        improved = False
        N = len(self.tour)
        for i in range(N - 1):
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue  # Do not reverse the entire tour
                # Calculate the change in distance
                a, b = (
                    self.cities_list[self.tour[i]],
                    self.cities_list[self.tour[i + 1]],
                )
                c, d = (
                    self.cities_list[self.tour[j]],
                    self.cities_list[self.tour[(j + 1) % N]],
                )
                delta = (
                    -math.hypot(a.x - b.x, a.y - b.y)
                    - math.hypot(c.x - d.x, c.y - d.y)
                    + math.hypot(a.x - c.x, a.y - c.y)
                    + math.hypot(b.x - d.x, b.y - d.y)
                )
                if delta < -1e-6:
                    # Perform the 2-opt swap
                    new_tour = (
                        self.tour[: i + 1]
                        + self.tour[i + 1 : j + 1][::-1]
                        + self.tour[j + 1 :]
                    )
                    self.tour = new_tour
                    self.best_distance += delta
                    improved = True
                    self.stagnant_iterations = 0  # Reset stagnation counter
                    self.draw_current_tour()
                    break
            if improved:
                break
        self.iteration += 1
        if improved:
            self.after(1, self.two_opt_iteration)
        else:
            self.optimizing = False
            print(
                "2-opt Optimization finished after {} iterations".format(self.iteration)
            )
            print("Best distance: {:.2f}".format(self.best_distance))

    # ACO-specific methods
    def initialize_pheromone(self):
        self.pheromone.clear()
        N = len(self.cities_list)
        initial_pheromone = 1.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    self.pheromone[(i, j)] = initial_pheromone

    def aco_iteration(self):
        if not self.optimizing or self.algorithm != "ACO":
            return
        N = len(self.cities_list)
        all_tours = []
        all_distances = []
        best_iteration_distance = None
        for ant in range(ant_count):
            tour = self.construct_solution()
            distance = self.calculate_tour_distance(tour)
            all_tours.append(tour)
            all_distances.append(distance)
            if self.best_distance is None or distance < self.best_distance:
                self.best_distance = distance
                self.tour = tour
                self.draw_current_tour()
                self.stagnant_iterations = 0  # Reset stagnation counter
            if best_iteration_distance is None or distance < best_iteration_distance:
                best_iteration_distance = distance
        else:
            self.stagnant_iterations += 1  # No improvement in this iteration
        self.update_pheromones(all_tours, all_distances)
        self.iteration += 1
        if (
            self.iteration < max_iterations
            and self.stagnant_iterations < stagnation_limit
        ):
            self.after(1, self.aco_iteration)
        else:
            self.optimizing = False
            print(
                "ACO Optimization finished after {} iterations".format(self.iteration)
            )
            print("Best distance: {:.2f}".format(self.best_distance))

    def construct_solution(self):
        N = len(self.cities_list)
        unvisited = set(range(N))
        current_city = random.choice(tuple(unvisited))
        tour = [current_city]
        unvisited.remove(current_city)
        while unvisited:
            probabilities = []
            total = 0.0
            for city in unvisited:
                tau = self.pheromone[(current_city, city)] ** alpha
                eta = (1.0 / self.distance(current_city, city)) ** beta
                prob = tau * eta
                probabilities.append((city, prob))
                total += prob
            # Normalize probabilities
            probabilities = [(city, prob / total) for city, prob in probabilities]
            # Choose next city based on probability
            rand = random.uniform(0, 1)
            cumulative = 0.0
            for city, prob in probabilities:
                cumulative += prob
                if cumulative >= rand:
                    next_city = city
                    break
            else:
                next_city = probabilities[-1][0]  # In case of rounding errors
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        return tour

    def update_pheromones(self, all_tours, all_distances):
        # Evaporate pheromones
        for key in self.pheromone:
            self.pheromone[key] *= 1 - evaporation
            if self.pheromone[key] < 1e-6:  # Avoid pheromone levels dropping to zero
                self.pheromone[key] = 1e-6
        # Deposit new pheromones
        for tour, distance in zip(all_tours, all_distances):
            pheromone_deposit = 1.0 / distance
            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone[(from_city, to_city)] += pheromone_deposit
                self.pheromone[
                    (to_city, from_city)
                ] += pheromone_deposit  # Assuming symmetric TSP

    # GA-specific methods
    def initialize_population(self):
        self.population.clear()
        N = len(self.cities_list)
        self.best_distance = None
        self.tour = None
        for _ in range(population_size):
            tour = list(range(N))
            random.shuffle(tour)
            distance = self.calculate_tour_distance(tour)
            if self.best_distance is None or distance < self.best_distance:
                self.best_distance = distance
                self.tour = tour
            self.population.append(tour)
        self.iteration = 0

    def ga_iteration(self):
        if not self.optimizing or self.algorithm != "GA":
            return
        new_population = []
        fitness_scores = []
        total_fitness = 0.0

        # Calculate fitness scores (inverse of distance)
        for tour in self.population:
            distance = self.calculate_tour_distance(tour)
            fitness = 1.0 / distance
            fitness_scores.append(fitness)
            total_fitness += fitness
            if distance < self.best_distance:
                self.best_distance = distance
                self.tour = tour[:]
                self.draw_current_tour()

        # Normalize fitness scores
        probabilities = [fitness / total_fitness for fitness in fitness_scores]

        # Selection and Crossover
        for _ in range(population_size // 2):
            parent1 = self.select_tour(probabilities)
            parent2 = self.select_tour(probabilities)
            if random.random() < crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            # Mutation
            if random.random() < mutation_rate:
                self.mutate(child1)
            if random.random() < mutation_rate:
                self.mutate(child2)
            new_population.extend([child1, child2])

        self.population = new_population
        self.iteration += 1
        if self.iteration < ga_iterations:
            self.after(1, self.ga_iteration)
        else:
            self.optimizing = False
            print("GA Optimization finished after {} iterations".format(self.iteration))
            print("Best distance: {:.2f}".format(self.best_distance))

    def select_tour(self, probabilities):
        rand = random.uniform(0, 1)
        cumulative = 0.0
        for tour, prob in zip(self.population, probabilities):
            cumulative += prob
            if cumulative >= rand:
                return tour
        return self.population[-1]  # In case of rounding errors

    def crossover(self, parent1, parent2):
        N = len(parent1)
        start, end = sorted(random.sample(range(N), 2))
        child1 = [None] * N
        child1[start : end + 1] = parent1[start : end + 1]
        pointer = 0
        for city in parent2:
            if city not in child1:
                while child1[pointer] is not None:
                    pointer += 1
                child1[pointer] = city

        child2 = [None] * N
        child2[start : end + 1] = parent2[start : end + 1]
        pointer = 0
        for city in parent1:
            if city not in child2:
                while child2[pointer] is not None:
                    pointer += 1
                child2[pointer] = city

        return child1, child2

    def mutate(self, tour):
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]

    def distance(self, city_index1, city_index2):
        a = self.cities_list[city_index1]
        b = self.cities_list[city_index2]
        return math.hypot(a.x - b.x, a.y - b.y)

    def draw_current_tour(self):
        if self.best_distance is None or self.tour is None:
            return  # Do not draw if no tour is available
        self.canvas.delete("tour")
        N = len(self.tour)
        for i in range(N):
            a = self.cities_list[self.tour[i]]
            b = self.cities_list[self.tour[(i + 1) % N]]
            self.canvas.create_line(
                a.x, a.y, b.x, b.y, fill="red", width=2, tags="tour"
            )
        # No need to redraw cities or edges; they are already drawn
        self.status_label.config(text="Distance: {:.2f}".format(self.best_distance))
        self.canvas.update()


if __name__ == "__main__":
    UI()