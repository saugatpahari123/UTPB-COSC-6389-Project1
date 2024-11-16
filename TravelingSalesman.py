import math
import random
import tkinter as tk
from tkinter import ttk


class City:
    """Represents a city in the TSP problem."""

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def draw(self, canvas, radius=8, color="#3498db"):
        """Draw the city as a circle on the canvas."""
        canvas.create_oval(
            self.x - radius, self.y - radius,
            self.x + radius, self.y + radius,
            fill=color, outline="white", tags="city"
        )


class TravelingSalesmanApp(tk.Tk):
    """Main application for solving the Traveling Salesman Problem."""

    def __init__(self):
        super().__init__()
        self.title("Traveling Salesman Problem Solver")
        self.geometry("1200x700")
        self.configure(bg="#ecf0f1")

        # Data
        self.cities = []
        self.tour = []
        self.best_distance = float("inf")
        self.is_solving = False

        # Layout
        self.control_frame = tk.Frame(self, bg="#2c3e50", width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        self.add_controls()

    def add_controls(self):
        """Add control buttons and labels to the control panel."""
        tk.Label(
            self.control_frame, text="TSP Solver", font=("Arial", 18, "bold"),
            bg="#2c3e50", fg="white", pady=20
        ).pack()

        ttk.Button(
            self.control_frame, text="Generate Cities", command=self.generate_cities
        ).pack(pady=10)

        ttk.Button(
            self.control_frame, text="Solve with 2-opt", command=self.solve_with_two_opt
        ).pack(pady=10)

        ttk.Button(
            self.control_frame, text="Reset", command=self.reset
        ).pack(pady=10)

        self.status_label = tk.Label(
            self.control_frame, text="Distance: 0.00", font=("Arial", 14),
            bg="#2c3e50", fg="white", pady=10
        )
        self.status_label.pack()

    def generate_cities(self):
        """Generate random cities."""
        self.reset()
        num_cities = 50
        padding = 50
        self.cities = [
            City(
                random.randint(padding, self.canvas.winfo_width() - padding),
                random.randint(padding, self.canvas.winfo_height() - padding),
                i
            )
            for i in range(num_cities)
        ]
        self.draw_cities()

    def draw_cities(self):
        """Draw all the cities on the canvas."""
        self.canvas.delete("city")
        for city in self.cities:
            city.draw(self.canvas)

    def reset(self):
        """Reset the application."""
        self.canvas.delete("all")
        self.cities.clear()
        self.tour.clear()
        self.best_distance = float("inf")
        self.status_label.config(text="Distance: 0.00")
        self.is_solving = False

    def solve_with_two_opt(self):
        """Solve the TSP using the 2-opt algorithm."""
        if not self.cities:
            print("No cities to solve. Please generate cities first.")
            return

        self.is_solving = True
        self.tour = list(range(len(self.cities)))
        self.best_distance = self.calculate_tour_distance(self.tour)
        self.perform_two_opt()

    def perform_two_opt(self):
        """Perform the 2-opt optimization."""
        if not self.is_solving:
            return

        improved = False
        for i in range(len(self.tour) - 1):
            for j in range(i + 2, len(self.tour)):
                if j == len(self.tour) - 1 and i == 0:
                    continue
                new_tour = self.tour[:i + 1] + self.tour[i + 1:j + 1][::-1] + self.tour[j + 1:]
                new_distance = self.calculate_tour_distance(new_tour)
                if new_distance < self.best_distance:
                    self.tour = new_tour
                    self.best_distance = new_distance
                    improved = True
                    self.draw_tour()
                    break
            if improved:
                break

        if improved:
            self.after(100, self.perform_two_opt)
        else:
            self.is_solving = False
            print(f"Optimization complete. Best distance: {self.best_distance:.2f}")

    def calculate_tour_distance(self, tour):
        """Calculate the total distance of a given tour."""
        distance = 0
        for i in range(len(tour)):
            a = self.cities[tour[i]]
            b = self.cities[tour[(i + 1) % len(tour)]]
            distance += math.hypot(a.x - b.x, a.y - b.y)
        return distance

    def draw_tour(self):
        """Draw the current best tour on the canvas."""
        self.canvas.delete("tour")
        for i in range(len(self.tour)):
            a = self.cities[self.tour[i]]
            b = self.cities[self.tour[(i + 1) % len(self.tour)]]
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill="#e74c3c", width=2, tags="tour")
        self.status_label.config(text=f"Distance: {self.best_distance:.2f}")


if __name__ == "__main__":
    app = TravelingSalesmanApp()
    app.mainloop()
