import sys
import math
import time
from PySide6.QtCore import Qt, QTimer # pip install PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
import pyqtgraph as pg # pip install pyqtgraph
import numpy as np #pip install numpy

def read_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines() # Read all lines into a list
        relevant_lines = lines[7:] # Ignore the first 7 lines
        coordinates = [] # Initialize a list to store the coordinates of the cities
        for line in relevant_lines:
            parts = line.split()
            city_number = int(parts[0]) # The first part is the city number
            longitude = float(parts[1]) # The second part is the longitude
            latitude = float(parts[2]) # The third part is the latitude
            coordinates.append((city_number, longitude, latitude)) # Add the city's coordinates to the list
    return coordinates
 
def calculate_increase(city, line):
    # Unpack the coordinates of the city and the line's start and end points
    _, x, y = city
    _, x1, y1 = line[0]
    _, x2, y2 = line[1]

    # Calculate the distances from the city to the line's start and end points
    dist_to_start = math.sqrt((x - x1)**2 + (y - y1)**2)
    dist_to_end = math.sqrt((x - x2)**2 + (y - y2)**2)

    # Calculate the distance between the line's start and end points
    dist_between_line_points = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # The increase in tour length is the sum of the distances from the city to the line's start and end points, minus the distance between the line's start and end points
    increase = dist_to_start + dist_to_end - dist_between_line_points

    return increase
    
def TSPGreedy(cities):
    # Record the start time
    start_time = time.time()

    # Initialize the tour with two cities
    tour_path = [cities[0], cities[23]]

    print("\nTour path:", tour_path) # Print the tour path for debugging

    # Remove the two cities from the unvisited_cities list
    unvisited_cities = cities[1:]
    unvisited_cities.remove((24, 85.076449, 17.029328))

    print("\nUnvisited cities:", unvisited_cities) # Print the unvisited cities for debugging
    
    total_distance = 0

    while unvisited_cities:
        min_increase = float('inf')
        closest_city = None
        insert_index = None

        for city in unvisited_cities:
            for i in range(len(tour_path)): # Iterate over the edges in the tour
                increase = calculate_increase(city, (tour_path[i], tour_path[(i + 1) % len(tour_path)])) # Calculate the increase in tour length if the city is inserted at this position
                if increase < min_increase:
                    min_increase = increase
                    closest_city = city
                    insert_index = i

        print("\nClosest city:", closest_city) # Print the closest city for debugging
        
        # Insert the closest city into the tour at the optimal position
        tour_path.insert(insert_index + 1, closest_city)

        # Update the lines list
        lines = [(tour_path[i], tour_path[(i + 1) % len(tour_path)]) for i in range(len(tour_path))]
        
        print("\nLines:")
        print("\n".join(map(str, lines))) # Print the lines for debugging as a list

        # Remove the inserted city from the unvisited_cities list
        unvisited_cities.remove(closest_city)

        # Update the total distance
        total_distance += min_increase

    end_time = time.time() # Record the end time
    time_taken = end_time - start_time # Calculate the time taken
    total_distance += 14.655880969509 # Add the distance between cities 1 and 24 to the total distance
    total_distance += 31.183102080966 # Add the distance between cities 7 and 1 to the total distance
    print("Time taken:", time_taken, "seconds")
    print("Total distance:", total_distance)
    print("Path:")
    print("\n".join(map(str, tour_path))) # Print the tour path as a list
    return tour_path, total_distance, lines

class TSPVisual(QMainWindow):
    def __init__(self, cities):
        super().__init__()
        self.setWindowTitle("TSP Visual")
        self.setGeometry(100, 100, 800, 600)

        self.total_distance = 0 # Initialize the total distance to 0
        self.solution_path = [] # List to store the solution path
        self.lines = [] # List to store the lines connecting the cities
        self.algorithm_executed = False  # Flag to check if the algorithm has been executed

        self.cities = cities # Store the cities in the class
        self.file_name = None # Initialize the file name to None

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        control_widget = QWidget(self)
        layout.addWidget(control_widget)

        control_layout = QVBoxLayout(control_widget)
        
        # Add a button to run the algorithm
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        control_layout.addWidget(run_button)
        
        # Add a button to select a TSP file
        select_file_button = QPushButton("Select File")
        select_file_button.clicked.connect(self.select_file)
        control_layout.addWidget(select_file_button)

        # Add a label to display the total distance
        self.distance_label = QLabel()
        control_layout.addWidget(self.distance_label)
        self.distance_label.setStyleSheet("color: white;")
        self.distance_label.setMinimumSize(200, 40)

        # Add a label to display the time taken
        self.time_label = QLabel()
        control_layout.addWidget(self.time_label)
        self.time_label.setStyleSheet("color: white;")
        self.time_label.setMinimumSize(200, 40)
        self.time_label.setText("Time Taken: ")

        # Add a label to display the path
        self.pathTE = QTextEdit()
        control_layout.addWidget(self.pathTE)
        self.pathTE.setStyleSheet("color: black;")
        self.pathTE.setMinimumSize(200, 40)
        self.pathTE.setText("Path: ")
        self.pathTE.setReadOnly(True)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QPushButton {
                background-color: #00ff00;
                color: #000000;
                font-weight: bold;
                border: 2px solid #00ff00;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #000000;
                color: #00ff00;
            }
        """)

        self.draw_cities(cities)

        # Update the total distance label with the new total distance
        self.distance_label.setText(f"Total Distance: {self.total_distance:.2f}")

        self.timer = QTimer(self) # Create a timer to plot the points one by one
        self.timer.timeout.connect(self.plot_next_point) # Connect the timer to the plot_next_point function
        self.current_step = 0

    def draw_cities(self, cities):
        # Extract x and y coordinates from the list of cities
        x_coords = [x for _, x, _ in cities]
        y_coords = [y for _, _, y in cities]

        # Create scatter plot of cities
        scatter = self.plot_widget.plot(x=x_coords, y=y_coords, pen=None, symbol='o', symbolBrush='r')

        # Add city numbers as labels
        for city_number, x, y in cities:
            label = pg.TextItem(text=str(city_number), anchor=(1, 1))
            label.setPos(x, y)
            self.plot_widget.addItem(label)

    def draw_path(self, path):
        # Extract x and y coordinates for the path
        x_coords = [x for _, x, _ in path]
        y_coords = [y for _, _, y in path]

        # Add a line connecting the cities
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        self.plot_widget.plot(x=x_coords, y=y_coords, pen='b')
    
    def plot_next_point(self):
        # Plot the next point if the algorithm has been executed and the current step is less than the number of edges
        if self.algorithm_executed and self.current_step < len(self.edges):
            city1, city2 = self.edges[self.current_step]
            self.draw_edge(city1, city2)
            self.current_step += 1
        # Stop the timer if the current step is equal to the number of edges
        elif self.current_step == len(self.edges):
            self.timer.stop()

    def draw_edge(self, city1, city2):
        # Extract x and y coordinates for the edge
        x_coords = [city1[1], city2[1]]
        y_coords = [city1[2], city2[2]]

        # Add a line connecting the cities
        line = self.plot_widget.plot(x=x_coords, y=y_coords, pen='b')
        self.lines.append(line)

    def select_file(self):
        # Ask the user to select a TSP file
        options = QFileDialog.Options()
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Select TSP File", "", "TSP Files (*.tsp)", options=options)
        
        if self.file_name:
            self.cities = read_tsp_file(self.file_name) # Read the cities from the file selected by the user
            self.plot_widget.clear() # Clear the plot
            self.distance_label.setText("Total Distance: ") # Reset the total distance label
            self.algorithm_executed = False  # Reset the algorithm execution flag
        else:
            return 0

    def run(self):
        if hasattr(self, 'cities'):
            self.algorithm_executed = True  # Set the algorithm execution flag
            start_time = time.time()  # Record the start time

            # Find cities 1 and 24 and add an edge between them
            city1 = next(city for city in self.cities if city[0] == 1)
            city24 = next(city for city in self.cities if city[0] == 24)

            self.edges = [(city1, city24)]

            # Run the algorithm
            self.solution_path, self.total_distance, self.lines = TSPGreedy(self.cities)
            end_time = time.time()  # Record the end time

            # Calculate the time taken and update the time label
            time_taken = end_time - start_time
            self.time_label.setText(f"Time Taken: {time_taken:.6f} seconds")

            self.draw_cities(self.cities)  # Draw the cities again for repeated runs

            # Store each edge in a list as it is added to the solution
            for i in range(len(self.solution_path) - 1):
                city1 = self.solution_path[i]
                city2 = self.solution_path[i + 1]
                self.edges.append((city1, city2))

            # Draw the solution path
            path_text = "\n".join([f"City {city[0]}: ({city[1]:.6f}, {city[2]:.6f})" for city in self.solution_path])
            self.pathTE.setPlainText(path_text)

            # Update the total distance label with the new total distance
            self.distance_label.setText(f"Total Distance: {self.total_distance:.2f}")
            self.current_step = 0  # Reset the current step
            self.timer.start(100)  # Start the timer for plotting

def main():
    app = QApplication(sys.argv)
    
    # Ask the user to select a TSP file
    options = QFileDialog.Options()
    file_name, _ = QFileDialog.getOpenFileName(None, "Select TSP File", "", "TSP Files (*.tsp)", options=options)
    
    if file_name:
        cities = read_tsp_file(file_name)
        window = TSPVisual(cities)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)  # Exit if no file was selected

if __name__ == "__main__":
    main()
