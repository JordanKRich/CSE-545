import random
import math
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSpinBox, QDoubleSpinBox, QComboBox
import pyqtgraph as pg
from pyqtgraph import TextItem
import sys

# Function to read the tsp file and return the coordinates of the cities
def read_tsp_file(filename):
    filename = "Random222.tsp"
    with open(filename, 'r') as file:
        # Open the file in read mode
        lines = file.readlines()
        relevant_lines = lines[7:]  # Discard the first 7 lines
        coordinates = []  # Initialize coordinates list
        for line in relevant_lines:
            # Read the relevant lines
            parts = line.split()
            # Split the line according to whitespaces
            city_number = int(parts[0])
            # First part is the city number
            longitude = float(parts[1])
            # Second part is the longitude
            latitude = float(parts[2])
            # Third part is the latitude
            coordinates.append((city_number, longitude, latitude))
            # Add the city and coordinates to the list
    return coordinates
# Function to initialize the population
def intialize_population(coordinates, population_size):
    #Initialize the population
    population = []

    #Get list of cities
    cities = [city[0] for city in coordinates]

    #End when all cities are visited
    for i in range(population_size):
        #Generate a random path
        path = random.sample(cities, len(cities))

        #Add the end city to the path
        path.append(path[0])

        #Add the path to the population
        population.append(path)
        print(f"Generated path: {i+1}")
    return population
# Function to calculate the distance between two cities
def calculate_distance(coord1, coord2):
    # Function to calculate the distance between two cities
    _, x1, y1 = coord1
    # Unpack the coordinates of the first city
    _, x2, y2 = coord2
    # Unpack the coordinates of the second city
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Calculate the Euclidean distance
    return distance
# Function to calculate the total distance of a path
def calculate_total_distance(path, coordinates):
    total_distance=0
    city_coordinates={city[0]: city for city in coordinates}
    #Calculate the total distance between each city in the path
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]
        coord1 = city_coordinates[city1]
        coord2 = city_coordinates[city2]

        # Print the path if it contains None values
        if None in path:
            print(f"path: {path}")
        if -1 in path:
            print(f"path: {path}")

        total_distance += calculate_distance(coord1, coord2)

    return total_distance
# Function to calculate the fitness of the path
def calculate_fitness(path, coordinates):
    #Calculate the fitness of the path
    total_distance=calculate_total_distance(path, coordinates)
    #The fitness is the inverse of the total distance
    fitness = 1 / total_distance
    return fitness
# Function to select parents using tournament selection
def select_parents_tournament(population, fitnesses, tournament_size):
    # Select `tournament_size` random paths from the population
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    # The path with the highest fitness wins the tournament
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]
# Function to perform edge recombination
def edge_recombination(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        # Create an adjacency list for each parent
        adjacency_list = {i: [] for i in parent1}
        for i in range(len(parent1)):
            # Add the neighbors of each city to the adjacency list
            if parent1[i-1] not in adjacency_list[parent1[i]]:
                adjacency_list[parent1[i]].append(parent1[i-1])
            if parent1[(i+1)%len(parent1)] not in adjacency_list[parent1[i]]:
                adjacency_list[parent1[i]].append(parent1[(i+1)%len(parent1)])
            if parent2[i-1] not in adjacency_list[parent2[i]]:
                adjacency_list[parent2[i]].append(parent2[i-1])
            if parent2[(i+1)%len(parent2)] not in adjacency_list[parent2[i]]:
                adjacency_list[parent2[i]].append(parent2[(i+1)%len(parent2)])
        # Create a child with the first city of parent1
        current_city = parent1[0]
        # Add the first city to the child
        child = [current_city]
        # Keep track of added cities in a set
        added_cities = set([current_city])  # Keep track of added cities in a set
        # Add the remaining cities to the child
        while len(child) < len(parent1):
            for _, neighbors in adjacency_list.items():
                if current_city in neighbors:
                    neighbors.remove(current_city)
            # Choose the next city
            if adjacency_list[current_city]:
                next_city = min(adjacency_list[current_city], key=lambda x: len(adjacency_list[x]))
            else:
                # If there are no neighbors, choose a random city
                remaining_cities = [city for city in parent1 if city not in added_cities]
                if remaining_cities:
                    next_city = random.choice(remaining_cities)
                else:
                    break
            # Add the next city to the child
            if next_city not in added_cities:
                current_city = next_city
                child.append(current_city)
                added_cities.add(current_city)
        
        child.append(child[0])
        child[-1] = child[0]
    else:
        child = parent1.copy()
        child[-1] = child[0]
    return child
# Function to perform ordered crossover
def crossover_ordered(parent1, parent2, crossover_rate):
    # Only perform crossover if a random number is less than the crossover rate
    if random.random() < crossover_rate:
        # Create a child with None values
        child = [None]*len(parent1)

        # Choose random start and end points for the slice
        start_pos = random.randint(1, len(parent1) - 2)  # Exclude first and last gene
        end_pos = random.randint(1, len(parent1) - 2)  # Exclude first and last gene

        # Ensure start_pos is less than end_pos
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos

        # Copy the slice from parent1 to child
        child[start_pos:end_pos] = parent1[start_pos:end_pos]

        # Fill in the remaining genes while preserving their order in parent2
        pointer_parent = pointer_child = 0
        while None in child: # While child contains None values
            if pointer_child >= len(child): # If we reach the end of the list
                pointer_child = 1  # Skip the first gene
            if pointer_parent >= len(parent2): # If we reach the end of the list
                pointer_parent = 0  # Wrap around to the start

            if parent2[pointer_parent] not in child: # If the gene isn't already in the child
                child[pointer_child] = parent2[pointer_parent] # Add the gene to the child
                pointer_child += 1
            pointer_parent += 1
            child[-1] = child[0]
    else:
        # If not performing crossover, just return parent1 (or parent2)
        child = parent1

    return child
# Function to perform inversion mutation
def mutation_inverse(path, mutation_rate):
    # Check if we should mutate
    if random.random() < mutation_rate:
        # Select two random indices such that i < j
        i, j = sorted(random.sample(range(1, len(path) - 1), 2))
        # Reverse the cities between i and j
        path[i:j] = reversed(path[i:j])
    return path
# Function to perform swap mutation
def mutation_swap(path, mutation_rate):
    # Go through each city in the path
    for i in range(1, len(path) - 1):
        # Check if we should mutate
        if random.random() < mutation_rate:
            # Select a random index to swap with
            j = random.randint(1, len(path) - 1)
            # Swap the cities
            path[i], path[j] = path[j], path[i]
    return path
# Function to check if the termination condition is met
def termination_condition(generation, max_generations=1000):
    # Check if the termination condition is met
    if generation >= max_generations:
        return True
    else:
        return False
# Function to get the best path from the population
def get_best_path(population, coordinates):
    # Get the best path from the population
    fitnesses = [calculate_fitness(path, coordinates) for path in population]
    # Get the index of the best path
    best_path_index = fitnesses.index(max(fitnesses))
    best_path=population[best_path_index]
    print (f"Best Path: {best_path}")
    return best_path
# Function to replace the population with the new population
def replace_population(population, new_population):
    #replace the population with the new population
    population = new_population
    return population
# Function to create the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Genetic Algorithm")

        #Creating widgets
        self.population_size_label = QLabel("Population Size")
        self.population_size_spinbox = QSpinBox()
        self.population_size_spinbox.setRange(1, 2000)
        self.population_size_spinbox.setValue(250)

        self.max_generations_label = QLabel("Max Generations")
        self.max_generations_spinbox = QSpinBox()
        self.max_generations_spinbox.setRange(1, 5000)
        self.max_generations_spinbox.setValue(500)

        self.mutation_rate_label = QLabel("Mutation Rate")
        self.mutation_rate_double_spinbox = QDoubleSpinBox()
        self.mutation_rate_double_spinbox.setRange(0, 1)
        self.mutation_rate_double_spinbox.setValue(0.1)

        self.elite_size_label = QLabel("Elite Size")
        self.elite_size_spinbox = QSpinBox()
        self.elite_size_spinbox.setRange(1, 100)
        self.elite_size_spinbox.setValue(50)

        self.tournament_size_label = QLabel("Tournament Size")
        self.tournament_size_spinbox = QSpinBox()
        self.tournament_size_spinbox.setRange(1, 100)
        self.tournament_size_spinbox.setValue(25)

        self.crossover_rate_label = QLabel("Crossover Rate")
        self.crossover_rate_double_spinbox = QDoubleSpinBox()
        self.crossover_rate_double_spinbox.setRange(0, 1)
        self.crossover_rate_double_spinbox.setValue(1)

        self.crossover_method_combobox = QComboBox()
        self.crossover_method_combobox.addItem("Edge Recombination")
        self.crossover_method_combobox.addItem("Ordered Crossover")

        self.mutation_method_combobox = QComboBox()
        self.mutation_method_combobox.addItem("Swap")
        self.mutation_method_combobox.addItem("Inverse")

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_genetic_algorithm)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setFixedHeight(200)

        self.path_plot_widget = pg.PlotWidget()
        self.path_plot_widget.setBackground("black")
        self.path_plot_widget.setFixedHeight(300)

        self.previous_button = QPushButton("Previous Path")
        self.previous_button.clicked.connect(self.show_previous_path)

        self.next_button = QPushButton("Next Path")
        self.next_button.clicked.connect(self.show_next_path)

        # Create a layout and add the widgets
        layout = QVBoxLayout()
        layout.addWidget(self.population_size_label)
        layout.addWidget(self.population_size_spinbox)
        layout.addWidget(self.max_generations_label)
        layout.addWidget(self.max_generations_spinbox)
        layout.addWidget(self.mutation_rate_label)
        layout.addWidget(self.mutation_rate_double_spinbox)
        layout.addWidget(self.elite_size_label)
        layout.addWidget(self.elite_size_spinbox)
        layout.addWidget(self.tournament_size_label)
        layout.addWidget(self.tournament_size_spinbox)
        layout.addWidget(self.crossover_rate_label)
        layout.addWidget(self.crossover_rate_double_spinbox)
        layout.addWidget(QLabel("Crossover Method"))
        layout.addWidget(self.crossover_method_combobox)
        layout.addWidget(QLabel("Mutation Method"))
        layout.addWidget(self.mutation_method_combobox)
        layout.addWidget(self.start_button)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.path_plot_widget)
        layout.addWidget(self.previous_button)
        layout.addWidget(self.next_button)

        # Create a widget and set the layout
        widget = QWidget()
        widget.setLayout(layout)

        # Set the widget as the central widget of the main window
        self.setCentralWidget(widget)
        self.current_path_index = 0

    def show_previous_path(self):
        # Decrement the current path index, ensuring it doesn't go below 0, and skipping 25 paths at a time
        self.current_path_index = max(0, self.current_path_index - 100)

        # Get the path at the current index
        path = self.best_paths[self.current_path_index]

        # Get the coordinates for each city in the path
        coords = [self.coordinates[city] for city in path]

        # Separate the x and y coordinates
        x_coords = [coord[1] for coord in coords]  # longitude
        y_coords = [coord[2] for coord in coords]  # latitude

        # Clear the plot and display the new path
        self.path_plot_widget.clear()
        self.path_plot_widget.setTitle(f"Generation: {self.current_path_index}")
        self.path_plot_widget.plot(x_coords, y_coords, pen="b")

        # Add city numbers to the plot
        for i, coord in enumerate(coords):
            text = pg.TextItem(str(coord[0]), anchor=(0.5, 0.5))
            self.path_plot_widget.addItem(text)
            text.setPos(coord[1], coord[2])
    
    def show_next_path(self):
        # Increment the current path index, ensuring it doesn't go above the number of paths, and skipping 25 paths at a time
        self.current_path_index = min(len(self.best_paths) - 1, self.current_path_index + 100)

        # Get the path at the current index
        path = self.best_paths[self.current_path_index]

        # Get the coordinates for each city in the path
        coords = [self.coordinates[city] for city in path]

        # Separate the x and y coordinates
        x_coords = [coord[1] for coord in coords]  # longitude
        y_coords = [coord[2] for coord in coords]  # latitude

        # Clear the plot and display the new path
        self.path_plot_widget.clear()
        self.path_plot_widget.setTitle(f"Generation: {self.current_path_index}")
        self.path_plot_widget.plot(x_coords, y_coords, pen="b")

        # Add city numbers to the plot
        for i, coord in enumerate(coords):
            text = pg.TextItem(str(coord[0]), anchor=(0.5, 0.5))
            self.path_plot_widget.addItem(text)
            text.setPos(coord[1], coord[2])

    def start_genetic_algorithm(self):
        # Get the parameters from the GUI
        population_size = self.population_size_spinbox.value()
        max_generations = self.max_generations_spinbox.value()
        mutation_rate = self.mutation_rate_double_spinbox.value()
        elite_size = self.elite_size_spinbox.value()
        tournament_size = self.tournament_size_spinbox.value()
        crossover_rate = self.crossover_rate_double_spinbox.value()
        crossover_method = self.crossover_method_combobox.currentText()
        mutation_method = self.mutation_method_combobox.currentText()

        # Read coordinates from .tsp file
        coordinates = read_tsp_file("Random222.tsp")
        self.coordinates = {coord[0]: coord for coord in coordinates}
        
        # Call TSPGA function and get the best path and all best paths
        best_path, self.best_paths = TSPGA("Random222.tsp", max_generations, population_size, mutation_rate, elite_size, tournament_size, crossover_rate, crossover_method, mutation_method, self.plot_widget)    
# Function to solve the TSP using GA
def TSPGA(filename, max_generations, population_size, initial_mutation_rate=0.01, elite_size=50, tournament_size=40,crossover_rate=1, crossover_method='Edge Recombination', mutation_method='Inverse', plot_widget=None ):
    coordinates = read_tsp_file(filename)
    # Initialize the population
    population = intialize_population(coordinates, population_size)
    generation=0
    best_distance=float("inf")
    mutation_rate=initial_mutation_rate
    best_distances=[]
    best_paths=[]
    total_distance_list = []  # Initialize an empty list to store all distances
    # Run the algorithm until the termination condition is met
    while not termination_condition(generation, max_generations):
        # Calculate the fitness of each path in the population
        fitnesses=[calculate_fitness(path, coordinates) for path in population]
        # Calculate the total distance of each path in the population
        total_distances=[calculate_total_distance(path, coordinates) for path in population]      
        total_distance_list.extend(total_distances)  # Add all distances to total_distance_list
        #calculate the average, median, and standard deviation of the total distances
        average= sum(total_distance_list)/len(total_distance_list)
        median= sorted(total_distance_list)[len(total_distance_list)//2]
        stdev= math.sqrt(sum([(x-average)**2 for x in total_distance_list])/len(total_distance_list))

        best_distances.append(min(total_distances))

        # Plot the best distance for each generation
        plot_widget.clear()
        plot_widget.plot(best_distances, pen="b")

        # Rank the population by fitness
        ranked_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
       
        #Sets the elite size to the integer value of the elite size
        elite_size = int(elite_size)

        # Create the next generation starting with the elite members of the current population
        new_population = ranked_population[:elite_size]

        # Add new children to the new population until it's the same size as the original population
        while len(new_population) < len(population):
            parent1 = select_parents_tournament(population, fitnesses, tournament_size)
            parent2 = select_parents_tournament(population, fitnesses, tournament_size)
            if crossover_method == 'Edge Recombination':
                child = edge_recombination(parent1, parent2, crossover_rate)
            elif crossover_method == 'Ordered Crossover':
                child = crossover_ordered(parent1, parent2, crossover_rate)
            if mutation_method == 'Swap':
                child = mutation_swap(child, mutation_rate)
            elif mutation_method == 'Inverse':
                child = mutation_inverse(child, mutation_rate)
            new_population.append(child)
        
        best_path_index = fitnesses.index(max(fitnesses))
        
        best_path_total_distance = total_distances[best_path_index]
        
        best_paths.append(population[best_path_index])
        
        print(f"Generation: {generation}\nBest Total Distance: {best_path_total_distance}\nBest Fitness: {max(fitnesses)}\nAverage: {average}\nMedian: {median}\nStandard Deviation: {stdev}\nMutation Rate: {mutation_rate}\n Best Path: {population[best_path_index]} ")
        
        generation += 1
        
        population = replace_population(population, new_population)

    return get_best_path(population, coordinates), best_paths  # Return total_distance_list
# Main function
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()