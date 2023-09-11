import sys
import math
import queue
import time
from PySide6.QtCore import Qt # pip install PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QTextEdit # pip install PySide6
import pyqtgraph as pg # pip install pyqtgraph
import numpy as np #pip install numpy


# Function to read the tsp file and return the coordinates of the cities
def read_tsp_file(filename):
    with open('11PointDFSBFS.tsp', 'r') as file:
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

def calculate_distance(coord1, coord2):
    # Function to calculate the distance between two cities
    _, x1, y1 = coord1
    # Unpack the coordinates of the first city
    _, x2, y2 = coord2
    # Unpack the coordinates of the second city
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Calculate the Euclidean distance
    return distance

# Define possible paths for each city
possible_paths = {
    0: [1, 2, 3],
    1: [2],
    2: [3, 4],
    3: [4, 5, 6],
    4: [6, 7],
    5: [7],
    6: [8, 9],
    7: [8, 9, 10],
    8: [10],
    9: [10]
}

def solve_tsp_bfs(coordinates):
    number_of_cities = len(coordinates)
    start_city = 0  # Start from the first city
    goal_city = number_of_cities - 1  # End at the last city
    visited = [False] * number_of_cities
    # Initialize the visited list to false
    q = queue.Queue()
    # Initialize the queue
   
    for next_city in possible_paths[start_city]:
        distance = calculate_distance(coordinates[start_city], coordinates[next_city])
        initial_path = [start_city, next_city]
        q.put((next_city, distance, initial_path))

    best_distance = float('inf')
    best_path = None
   
    while not q.empty():
        current_city, current_distance, current_path = q.get()

        if current_city == goal_city:
            if current_distance < best_distance:
                best_distance = current_distance
                best_path = current_path
            continue

        visited[current_city] = True

        next_cities = possible_paths.get(current_city, [])
        print("Current City:", current_city + 1)
        print("Next Cities:", next_cities)
        print("Visited:", visited)
        print("\n")

        for next_city in next_cities:
            if not visited[next_city]:
                distance = calculate_distance(coordinates[current_city], coordinates[next_city])
                next_path = current_path + [next_city]
                q.put((next_city, current_distance + distance, next_path))
    
    print("Visited:", visited)
    print("Best Distance:", best_distance)
    print("Best Path:", [city + 1 for city in best_path])
    print("Current City:", current_city + 1)

    return best_distance, best_path

def solve_tsp_dfs(coordinates):
    number_of_cities = len(coordinates)
    start_city = 0
    goal_city = number_of_cities - 1

    visited = [False] * number_of_cities # Initialize the visited list to false so that we can keep track of the visited cities
    current_city = start_city
    current_path = [start_city]
    current_distance = 0
    best_distance = float('inf')
    best_path = None

    def dfs(current_city, current_distance, current_path):
        nonlocal best_distance, best_path # Use the nonlocal keyword to access the variables in the outer scope

        if current_city == goal_city:
            if current_distance < best_distance:
                best_distance = current_distance
                best_path = current_path.copy()  # Make a copy of the current path
            return

        visited[current_city] = True
        next_cities = possible_paths[current_city]
        print("Next Cities:", next_cities)
        print("Visited:", visited)
        print("Current City:", current_city + 1)
        print("Current Path:", current_path)
        print("\nBest Path", best_path)
        for next_city in next_cities:
            if not visited[next_city]:
                distance = calculate_distance(coordinates[current_city], coordinates[next_city])
                current_path.append(next_city)
                current_distance += distance
                dfs(next_city, current_distance, current_path)
                current_path.pop()  # Revert the current path
                current_distance -= distance

        visited[current_city] = False

    dfs(current_city, current_distance, current_path)
    return best_distance, best_path

class TSPSolverApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TSP Solver")
        self.setGeometry(100, 100, 750, 330)
        self.init_ui()

    def init_ui(self):
        self.label = QLabel("Select an algorithm:", self)
        self.label.setGeometry(25, 25, 200, 50)

        self.algorithim_combo_box = QComboBox(self)
        self.algorithim_combo_box.addItems(["BFS", "DFS"])
        self.algorithim_combo_box.setGeometry(50, 75, 200, 50)

        self.solve_button = QPushButton("Solve", self)
        self.solve_button.setGeometry(50, 125, 200, 50)
        self.solve_button.clicked.connect(self.solve_and_plot_tsp)

        self.result_text_edit = QTextEdit(self)
        self.result_text_edit.setGeometry(50, 175, 200, 100)
        self.result_text_edit.setReadOnly(True)

        self.plot_widget = pg.GraphicsLayoutWidget(self)
        self.plot_widget.setGeometry(300, 40, 400, 250)

        self.plot_view = self.plot_widget.addViewBox()
        self.plot_view.setAspectLocked(True)

        self.plot = self.plot_widget.addPlot(0,0,200,400)
        self.plot.showGrid(x=True, y=True)

    def plot_graph(self, x_data, y_data, solution_path=None):
        self.plot.clear()

        all_cities_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255))
        all_cities_scatter.addPoints(x=x_data, y=y_data)
        self.plot.addItem(all_cities_scatter)

        if solution_path:
            # Convert the solution path to 0-based indexing for processing
            path_x = [x_data[city] for city in solution_path]
            path_y = [y_data[city] for city in solution_path]
            
            path_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
            path_scatter.addPoints(x=path_x, y=path_y)

            # Create lines between cities in the solution path (white)
            for i in range(len(solution_path)-1):
                start_city = solution_path[i]
                end_city = solution_path[i+1]
                line = pg.PlotCurveItem(x=[x_data[start_city], x_data[end_city]],
                                        y=[y_data[start_city], y_data[end_city]],
                                        pen=pg.mkPen('white'))
                self.plot.addItem(line)

            self.plot.addItem(path_scatter)

        # Add labels with city numbers next to each city point
        for i, (x, y) in enumerate(zip(x_data, y_data), start=1):
            label = pg.TextItem(text=str(i), anchor=(0.5, 0.5))
            label.setPos(x, y)
            self.plot.addItem(label)

        self.plot_view
        
    def solve_and_plot_tsp(self):
        algorithim = self.algorithim_combo_box.currentText()
        filename = '11PointDFSBFS.tsp'  # Replace with your TSP file
        coordinates = read_tsp_file(filename)

        if coordinates is not None:
            
            start_time = time.time()

            if algorithim == "BFS":
                result, path = solve_tsp_bfs(coordinates)
            elif algorithim == "DFS":
                result, path = solve_tsp_dfs(coordinates)
            else:
                self.result_text_edit.setText("Invalid algorithm selected.")
                return
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            result_text = f"{algorithim} Shortest Distance: {result}\n" # I used an f-string here because it's easier to format the text
            result_text += f"{algorithim} Shortest Path: {[city + 1 for city in path]}\n"
            result_text += f"Elapsed Time: {elapsed_time:.4f} seconds"
            self.result_text_edit.setPlainText(result_text)

            # Extract the x and y coordinates from the coordinates list
            x_data = [coord[1] for coord in coordinates]
            print(x_data)
            y_data = [coord[2] for coord in coordinates]
            print(y_data)  

            # Plot the graph with the solution path
            self.plot_graph(x_data, y_data, solution_path=path)
            
        else:
            self.result_text_edit.setText("No solution found!")

def main():
    app = QApplication(sys.argv)
    window = TSPSolverApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
