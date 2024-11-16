import math
import random
import tkinter as tk
from tkinter import *
import threading

num_items = 100
frac_target = 0.7
min_value = 128
max_value = 2048

screen_padding = 25
item_padding = 5
stroke_width = 5

num_generations = 1000
pop_size = 50
elitism_count = 2
mutation_rate = 0.1

sleep_time = 0.1


def random_rgb_color():
    red = random.randint(0x10, 0xff)
    green = random.randint(0x10, 0xff)
    blue = random.randint(0x10, 0xff)
    hex_color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
    return hex_color


class Item:
    def __init__(self):
        self.value = random.randint(min_value, max_value)
        self.color = random_rgb_color()
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def place(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, canvas, active=False):
        canvas.create_text(self.x+self.w+item_padding+stroke_width*2, self.y+self.h/2, text=f'{self.value}')
        if active:
            canvas.create_rectangle(self.x,
                                    self.y,
                                    self.x+self.w,
                                    self.y+self.h,
                                    fill=self.color,
                                    outline=self.color,
                                    width=stroke_width)
        else:
            canvas.create_rectangle(self.x,
                                    self.y,
                                    self.x+self.w,
                                    self.y+self.h,
                                    fill='',
                                    outline=self.color,
                                    width=stroke_width)


class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        # Set the title of the window
        self.title("Knapsack")
        # Hide the minimize/maximize/close decorations at the top of the window frame
        #   (effectively making it act like a full-screen application)
        self.option_add("*tearOff", FALSE)
        # Get the screen width and height
        self.width, self.height = self.winfo_screenwidth(), self.winfo_screenheight()
        # Set the window width and height to fill the screen
        self.geometry("%dx%d+0+0" % (self.width, self.height))
        # Set the window content to fill the width * height area
        self.state("zoomed")

        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=self.width, height=self.height)

        self.items_list = []

        # We create a standard banner menu bar and attach it to the window
        menu_bar = Menu(self)
        self['menu'] = menu_bar

        # We have to individually create the "File", "Edit", etc. cascade menus, and this is the first
        menu_K = Menu(menu_bar)
        # The underline=0 parameter doesn't actually do anything by itself,
        #   but if you also create an "accelerator" so that users can use the standard alt+key shortcuts
        #   for the menu, it will underline the appropriate key to indicate the shortcut
        menu_bar.add_cascade(menu=menu_K, label='Knapsack', underline=0)

        def generate():
            self.generate_knapsack()
            self.draw_items()
        # The add_command function adds an item to a menu, as opposed to add_cascade which adds a sub-menu
        # Note that we use command=generate without the () - we're telling it which function to call,
        #   not actually calling the function as part of the add_command
        menu_K.add_command(label="Generate", command=generate, underline=0)

        self.target = 0

        def set_target():
            target_set = []
            for x in range(int(num_items * frac_target)):
                item = self.items_list[random.randint(0, len(self.items_list)-1)]
                while item in target_set:
                    item = self.items_list[random.randint(0, len(self.items_list) - 1)]
                target_set.append(item)
            total = 0
            for item in target_set:
                total += item.value
            self.target = total
            self.draw_target()
        menu_K.add_command(label="Get Target", command=set_target, underline=0)

        def start_thread():
            thread = threading.Thread(target=self.run, args=())
            thread.start()
        menu_K.add_command(label="Run", command=start_thread, underline=0)

        # We have to call self.mainloop() in our constructor (__init__) to start the UI loop and display the window
        self.mainloop()

    def get_rand_item(self):
        i1 = Item()
        for i2 in self.items_list:
            if i1.value == i2.value:
                return None
        return i1

    def add_item(self):
        item = self.get_rand_item()
        while item is None:
            item = self.get_rand_item()
        self.items_list.append(item)

    def generate_knapsack(self):
        for i in range(num_items):
            self.add_item()

        item_max = 0
        item_min = 9999
        for item in self.items_list:
            item_min = min(item_min, item.value)
            item_max = max(item_max, item.value)

        w = self.width - screen_padding
        h = self.height - screen_padding
        num_rows = math.ceil(num_items / 6)
        row_w = w / 8 - item_padding
        row_h = (h - 200) / num_rows
        # print(f'{w}, {h}, {num_rows}, {row_w}, {row_h}')
        for x in range(0, 6):
            for y in range(0, num_rows):
                if x * num_rows + y >= num_items:
                    break
                item = self.items_list[x * num_rows + y]
                item_w = row_w / 2
                item_h = max(item.value / item_max * row_h, 1)
                # print(f'{screen_padding+x*row_w+x*item_padding},'
                #      f'{screen_padding+y*row_h+y*item_padding},'
                #      f'{item_w},'
                #      f'{item_h}')
                item.place(screen_padding + x * row_w + x * item_padding,
                           screen_padding + y * row_h + y * item_padding,
                           item_w,
                           item_h)

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw_items(self):
        for item in self.items_list:
            item.draw(self.canvas)

    def draw_target(self):
        x = (self.width - screen_padding) / 8 * 7
        y = screen_padding
        w = (self.width - screen_padding) / 8 - screen_padding
        h = self.height / 2 - screen_padding
        self.canvas.create_rectangle(x, y, x + w, y + h, fill='black')
        self.canvas.create_text(x+w//2, y+h+screen_padding, text=f'{self.target}', font=('Arial', 18))

    def draw_sum(self, item_sum, target):
        x = (self.width - screen_padding) / 8 * 6
        y = screen_padding
        w = (self.width - screen_padding) / 8 - screen_padding
        h = self.height / 2 - screen_padding
        # print(f'{item_sum} / {target} * {h} = {item_sum/target} * {h} = {item_sum/target*h}')
        h *= (item_sum / target)
        self.canvas.create_rectangle(x, y, x + w, y + h, fill='black')
        self.canvas.create_text(x+w//2, y+h+screen_padding, text=f'{item_sum} ({"+" if item_sum>target else "-"}{abs(item_sum-target)})', font=('Arial', 18))

    def draw_genome(self, genome, gen_num):
        for i in range(num_items):
            item = self.items_list[i]
            active = genome[i]
            item.draw(self.canvas, active)
        x = (self.width - screen_padding) / 8 * 6
        y = screen_padding
        w = (self.width - screen_padding) / 8 - screen_padding
        h = self.height / 4 * 3
        self.canvas.create_text(x + w, y + h + screen_padding*2, text=f'Generation {gen_num}', font=('Arial', 18))

    def run(self):
        global pop_size
        global num_generations

        def gene_sum(genome):
            total = 0
            for i in range(len(genome) - 1):
                if genome[i]:
                    total += self.items_list[i].value
            return total

        def fitness(genome):
            return abs(gene_sum(genome) - self.target)

        def get_population(last_pop=None, fitnesses=None):
            population = []
            if last_pop is None:
                for g in range(pop_size):
                    genome = []
                    for bit in range(num_items):
                        genome.append(random.random() < frac_target)
                    population.append(genome)
                return population
            else:
                # elitism
                elites = []
                for e in range(elitism_count):
                    elites.append(fitnesses[e])
                for e in last_pop:
                    if fitness(e) in elites:
                        population.append(e)

                def select_parents(min_fitness):
                    weights = []
                    for parent in last_pop:
                        if fitness(parent) == 0.0:
                            weights.append(1.0)
                        else:
                            weights.append(min_fitness / fitness(parent))

                    def get_by_weight():
                        idx = random.randint(0, pop_size - 1)
                        while random.random() < weights[idx]:
                            idx = random.randint(0, pop_size - 1)
                        return last_pop[idx]

                    return get_by_weight(), get_by_weight()

                def crossover(parent1, parent2):
                    length = len(parent1)
                    x = random.randint(0, length // 2)
                    y = x + length // 2
                    g_out = []
                    for i in range(length):
                        if x < i <= y:
                            g_out.append(parent2[i])
                        else:
                            g_out.append(parent1[i])
                    if len(g_out) < num_items:
                        print('Error!')
                    return g_out

                def mutate(g_in):
                    x = random.randint(0, len(g_in) - 1)
                    g_out = []
                    for i in range(len(g_in)):
                        if i == x:
                            g_out.append(not g_in[i])
                        else:
                            g_out.append(g_in[i])
                    return g_out

                # fill generation with new individuals
                while len(population) < pop_size:
                    # select two random parents by weighted selection
                    # note no guarantee of uniqueness - could get the same parent twice
                    parents = select_parents(fitnesses[0])
                    # perform crossover to generate new individual
                    baby = crossover(parents[0], parents[1])
                    # potentially perform mutation
                    if random.random() < mutation_rate:
                        baby = mutate(baby)
                    # add to next generation
                    population.append(baby)

                return population

        def generation_step(generation=0, pop=None):
            if generation >= num_generations:
                return  # Stop the process after the set number of generations

            if pop is None:
                pop = get_population()

            fitnesses = []
            best_of_gen = None
            min_fitness = 9999
            for genome in pop:
                fit = fitness(genome)
                if fit < min_fitness:
                    best_of_gen = genome
                    min_fitness = fit
                fitnesses.append(fit)
            fitnesses.sort()

            print(f'Best fitness of generation {generation}: {min_fitness}')
            print(best_of_gen)
            print()

            # Schedule the UI updates in the main thread
            self.after(0, self.clear_canvas)
            self.after(0, self.draw_target)
            self.after(0, self.draw_sum, gene_sum(best_of_gen), self.target)
            self.after(0, self.draw_genome, best_of_gen, generation)

            # Schedule the next generation step after a delay, unless we're at the global optimum (fitness == 0)
            if fitnesses[0] != 0:
                self.after(int(sleep_time * 1000), generation_step, generation + 1, get_population(pop, fitnesses))

        # Start the evolutionary process
        generation_step()


# In python, we have this odd construct to catch the main thread and instantiate our Window class
if __name__ == '__main__':
    UI()