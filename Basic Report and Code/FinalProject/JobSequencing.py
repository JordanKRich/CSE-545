import random
import copy
from PySide6.QtWidgets import QApplication, QMainWindow, QSpinBox, QLabel, QPushButton, QGridLayout, QWidget, QLineEdit, QFileDialog, QMessageBox, QDoubleSpinBox, QTextEdit
import pyqtgraph as pg
import time
import os
import pandas as pd

#Vairables for GUI
mutation_rate = 0.1
crossover_rate = 0.7
population_size = 100
generations = 100
elitism_count = 2
tournament_size=3
number_of_runs = 5
# Variables for storing data
all_runs_results = []
total_times = []
total_times_and_generation = []
best_woc_population = []
top_solutions_global = []
woc_best= []
woc_best_fitness=0
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the 'generated_instance.txt' file
filename = os.path.join(script_dir, "generated_instance.txt")
# Function to create a GUI for the user to interact with
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__() 
        # Initialize the graph widget
        self.graphWidget = pg.PlotWidget()

        # Initialize the text items list
        self.textItems = []

        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        # Place the graph widget on the layout
        layout.addWidget(self.graphWidget, 0, 0, 1, 4)

        # Initialize QLabel for GA path and fitness
        self.gaPathLabel = QTextEdit("Best GA Path: Not computed yet")
        layout.addWidget(self.gaPathLabel, 11, 0, 1, 4)

        # Initialize QLabel for WoC path and fitness
        self.wocPathLabel = QTextEdit("Best WoC Path: Not computed yet")
        layout.addWidget(self.wocPathLabel, 12, 0, 1, 4)

        # Mutation rate input
        self.mutation_rate_input = QDoubleSpinBox()
        self.mutation_rate_input.setRange(0, 1)
        self.mutation_rate_input.setValue(mutation_rate)
        self.mutation_rate_input.valueChanged.connect(self.update_mutation_rate)
        layout.addWidget(QLabel("Mutation Rate:"), 1, 0)
        layout.addWidget(self.mutation_rate_input, 1, 1)

        # Elitism count input
        self.elitism_count_input = QSpinBox()
        self.elitism_count_input.setRange(1, population_size)
        self.elitism_count_input.setValue(elitism_count)
        self.elitism_count_input.valueChanged.connect(self.update_elitism_count)
        layout.addWidget(QLabel("Elitism Count:"), 5, 0)
        layout.addWidget(self.elitism_count_input, 5, 1)
        
        # Tournament size input
        self.tournament_size_input = QSpinBox()
        self.tournament_size_input.setRange(1, population_size)
        self.tournament_size_input.setValue(tournament_size)
        self.tournament_size_input.valueChanged.connect(self.update_tournament_size)
        layout.addWidget(QLabel("Tournament Size:"), 6, 0)
        layout.addWidget(self.tournament_size_input, 6, 1)
        
        # Number of runs
        self.number_of_runs_input = QSpinBox()
        self.number_of_runs_input.setRange(1, 20)
        self.number_of_runs_input.setValue(5)
        self.number_of_runs_input.valueChanged.connect(self.update_number_of_runs)
        layout.addWidget(QLabel("Number of Runs:"), 7, 0)
        layout.addWidget(self.number_of_runs_input, 7, 1)
        
        # Crossover rate input
        self.crossover_rate_input = QDoubleSpinBox()
        self.crossover_rate_input.setRange(0, 1)
        self.crossover_rate_input.setValue(crossover_rate)
        self.crossover_rate_input.valueChanged.connect(self.update_crossover_rate)
        layout.addWidget(QLabel("Crossover Rate:"), 2, 0)
        layout.addWidget(self.crossover_rate_input, 2, 1)

        # Population size input
        self.population_size_input = QSpinBox()
        self.population_size_input.setRange(1, 1000)
        self.population_size_input.setValue(population_size)
        self.population_size_input.valueChanged.connect(self.update_population_size)
        layout.addWidget(QLabel("Population Size:"), 3, 0)
        layout.addWidget(self.population_size_input, 3, 1)

        # Generations input
        self.generations_input = QSpinBox()
        self.generations_input.setRange(1, 1000)
        self.generations_input.setValue(generations)
        self.generations_input.valueChanged.connect(self.update_generations)
        layout.addWidget(QLabel("Generations:"), 4, 0)
        layout.addWidget(self.generations_input, 4, 1)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_algorithm)
        layout.addWidget(self.run_button, 8, 0, 1, 2)

        # File selection input and button
        self.file_input = QLineEdit(self)
        self.file_input.setPlaceholderText("Path to file...")
        layout.addWidget(self.file_input, 9, 0, 1, 3)
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button, 9, 3)
        
        #Previous Run
        self.prev_button=QPushButton("Previous Run")
        self.prev_button.clicked.connect(self.prev_run)
        layout.addWidget(self.prev_button, 10, 0, 1, 2)
        #Next Run
        self.next_button=QPushButton("Next Run")
        self.next_button.clicked.connect(self.next_run)
        layout.addWidget(self.next_button, 10, 2, 1, 2)
        
    def update_mutation_rate(self, value):
        global mutation_rate
        mutation_rate = value
        
    def update_elitism_count(self, value):
        global elitism_count
        elitism_count = value
        
    def update_number_of_runs(self, value):
        global number_of_runs
        number_of_runs = value
        
    def update_tournament_size(self, value):
        global tournament_size
        tournament_size = value    
        
    def update_crossover_rate(self, value):
        global crossover_rate
        crossover_rate = value

    def update_population_size(self, value):
        global population_size
        population_size = value

    def update_generations(self, value):
        global generations
        generations = value
    
    def prev_run(self):
        if self.current_run_index>0:
            self.current_run_index-=1
            self.graphWidget.clear()
            self.update_graph(all_runs_results[self.current_run_index])
    
    def next_run(self):
        if self.current_run_index<len(all_runs_results)-1:
            self.current_run_index+=1
            self.graphWidget.clear()
            self.update_graph(all_runs_results[self.current_run_index])

    def select_file(self):
        # Open file dialog and set the selected file path to the input
        file_path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if file_path:
            self.file_input.setText(file_path)

    def run_algorithm(self):
        file_path = self.file_input.text()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please select a file first.")
            return
        
        # Clear previous data
        global total_times, total_times_and_generation, all_runs_results
        all_runs_results = []
        total_times = []
        total_times_and_generation = []
        tsp_solutions_global = []
        
        self.graphWidget.clear()
        jobs = read_jobs_from_file(file_path)
        self.run_button.setEnabled(False)
        
        best_solutions = []
        for run in range(number_of_runs):
            print(f"Run {run + 1} of {number_of_runs}")
            population = generate_initial_population(jobs)
            best_individual = GA(jobs, population, population_size=population_size, generations=generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elitism_count=elitism_count, tournament_size=tournament_size)
            best_solutions.append(best_individual)
            all_runs_results.append(total_times_and_generation.copy())
            total_times.clear()
            total_times_and_generation.clear()
        
        woc_solution = create_woc_solution(best_solutions)

        for job in woc_solution:
            if not isinstance(job, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in job):
                raise ValueError("WoC solution structure is incorrect.")
        # Run GA with WoC population
        woc_population = initialize_woc_population(jobs, woc_solution, population_size)
        for individual in woc_population:
            for job in individual:
                if not isinstance(job, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in job):
                    raise ValueError("WoC population structure is incorrect.")
        woc_best_individual = GA(jobs, woc_population, population_size=population_size, generations=generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elitism_count=elitism_count, tournament_size=tournament_size)
        sorted_best_solutions = sorted(best_solutions, key=lambda solution: fitness(solution))
        best_ga_solution = sorted_best_solutions[0]
        self.gaPathLabel.setText(f"Best GA Path: {best_ga_solution}\nFitness: {fitness(best_ga_solution)}")
        self.wocPathLabel.setText(f"Best WOC Path: {woc_best_individual}\nFitness: {fitness(woc_best_individual)}")
        self.run_button.setEnabled(True)
        self.woc_best_fitness=fitness(woc_best_individual)
        self.current_run_index = 0
        self.update_graph(all_runs_results[self.current_run_index])
        print("Best individual from standard GA: ", best_individual)
        print("Best fitness from standard GA: ", fitness(best_individual))
        print("Best individual from WoC GA: ", woc_best_individual)
        print("Best fitness from WoC GA: ", fitness(woc_best_individual))
        print(f"Completed {number_of_runs} standard GA runs and 1 WoC GA run\n")
        df = pd.DataFrame({'Run': range(1, len(best_solutions) + 1),
                       'Fitness': [fitness(solution) for solution in best_solutions]})

        # Export to Excel
        excel_filename = "best_solutions.xlsx"
        df.to_excel(excel_filename, index=False)

        print(f"Best solutions exported to {excel_filename}\n")
        
    def update_graph(self, total_times_and_generation):
        woc_data_point = (max(total_times_and_generation, default=(0, 0))[0] + 1, self.woc_best_fitness)
        filtered_data = [(gen, fitness) for gen, fitness in total_times_and_generation if fitness is not None]
        genx = [item[0] for item in filtered_data]
        fitnesses = [item[1] for item in filtered_data]
        if len(genx) == len(total_times_and_generation):
            genx.append(woc_data_point[0])
            fitnesses.append(woc_data_point[1])

        symbolIndexes = [i for i in range(len(genx)) if i == (len(genx)-1) or i == (len(genx)-2) or i == 0]  # Adjusted condition here

        symbols = ['star' if i in symbolIndexes else None for i in range(len(genx))]  # List of symbols for each point

        self.graphWidget.plot(genx, fitnesses, pen='blue', symbol=symbols, symbolPen='purple', symbolBrush='white', name='Fitness over Generations')

        # Remove old text items and clear the list
        for textItem in self.textItems:
            self.graphWidget.removeItem(textItem)
        self.textItems.clear()

        # Create new text items only for marked points
        for i in symbolIndexes:
            x, y = genx[i], fitnesses[i]
            text = pg.TextItem(text=f"{y}", anchor=(0.5, 0))
            self.graphWidget.addItem(text)
            text.setPos(x, y)
            self.textItems.append(text)
# Function to read the jobs from the file
def read_jobs_from_file(filename):
    jobs = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[5:]:
            numbers = list(map(int, line.split()))
            job = [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]
            jobs.append(job)
    return jobs
# Function to generate the initial population
def generate_initial_population(jobs):
    population = []
    for _ in range(population_size):
        random_jobs = copy.deepcopy(jobs)  # create a deep copy of jobs
        random.shuffle(random_jobs)  # randomize the order of jobs
        population.append(random_jobs)  # add the randomized jobs to the population
    return population
# Function to mutate the child
def mutate(child):
    num_swaps = random.randint(1, len(child) // 2)
    for _ in range(num_swaps):
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
    # Ensure the child is a new individual if mutation rate is low
    if num_swaps == 0:
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
# Function to select the fittest individual
def tournament_selection(population, tournament_size): #Tournament Selection
    participants = random.sample(population, tournament_size)
    fittest_individual = min(participants, key=fitness)
    return fittest_individual
# Function to perform crossover
def crossover(parent1, parent2):
    child = []
    added_jobs = set()
    for gene1, gene2 in zip(parent1, parent2):
        gene1_tuple = tuple(gene1)  # Convert list to tuple
        gene2_tuple = tuple(gene2)
        if gene1_tuple not in added_jobs:
            child.append(gene1)
            added_jobs.add(gene1_tuple)
        elif gene2_tuple not in added_jobs:
            child.append(gene2)
            added_jobs.add(gene2_tuple)

    # In case both jobs in gene pair were already added, fill in missing jobs
    missing_jobs = [job for job in parent1 + parent2 if tuple(job) not in added_jobs]
    for job in missing_jobs:
        if len(child) < len(parent1):  # Still space left in the child
            child.append(job)
            added_jobs.add(tuple(job))  # Add as tuple to the set
    
    return child
# Get fitness of each schedule 
def fitness(schedule):
    num_machines = len(schedule[0])
    job_end_times = [0] * len(schedule)  # end times of jobs
    machine_end_times = [0] * num_machines  # end times on each machine

    for job in schedule:
        last_end_time = 0  # to keep track of when the last task of this job ended
        for machine, time in job:
            # Calculate the start time of this task: it can't start before the previous task of this job ended,
            # and it can't start before the last task on this machine ended
            start_time = max(last_end_time, machine_end_times[machine])
            end_time = start_time + time  # When the task will be finished

            last_end_time = end_time
            machine_end_times[machine] = end_time
            job_end_times[schedule.index(job)] = end_time  # update end time of this job

    makespan = max(job_end_times)  # the time when the last job is finished
    total_times.append(makespan)
    return makespan
# Genetic Algorithm
def GA(jobs, population, generations, population_size, crossover_rate, mutation_rate, elitism_count, tournament_size):
    start_time = time.time()
    best_fitnesses = []

    for gen in range(generations):
        fitness_values = [fitness(ind) for ind in population]  # Calculate fitness for all
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])
        population = [population[i] for i in sorted_indices]
        
        best_fitness= fitness_values[sorted_indices[0]]
        
        print(f"Generation: {gen}\n Best Fitness: {best_fitness}\n")
        best_fitnesses.append(fitness_values[sorted_indices[0]])
        
        
        total_times_and_generation.append((gen, fitness_values[sorted_indices[0]]))
        
        # number of top individuals to retain
        new_population = population[:elitism_count]
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)  # Tournament selection
            parent2 = tournament_selection(population, tournament_size)  # Tournament selection
            
            if random.random() < crossover_rate:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
                
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]  # Ensure we don't exceed population size due to elitism and offspring
        
    print("Time taken: ", time.time() - start_time, "seconds\n")
    print(f"Total Times and Generation: {total_times_and_generation}\n")
    return min(population, key=fitness)
# Repairing Algorithm 
def repair_greedy(jobs):
    # TODO: Implement greedy solution here for repairing route if broken
    pass
# Main Control for doing computations 
def main_solver(jobs, generations, population_size, crossover_rate, mutation_rate):
    population = generate_initial_population(jobs)
    total_times_and_generation = []
    best_indivual = GA(jobs, population)
    print("Best individual: ", best_indivual)
    repair_greedy(jobs)
    if len(total_times_and_generation) == generations - 1:
        print(f"Total Times and Generation: {total_times_and_generation}")
    return total_times_and_generation
# Function to store the top solutions
def store_top_solutions(solutions, percentage_of_solutions=0.20):
    number_to_keep=int(len(solutions)*percentage_of_solutions)
    
    sorted_solutions=sorted(solutions, key=fitness, reverse=True)
    
    top_solutions=sorted_solutions[:number_to_keep]
    
    return top_solutions
# Function to initialize the WoC population
def initialize_woc_population(jobs, woc_solution, population_size):
    woc_population = []
    # Add the woc_solution directly
    woc_population.append(woc_solution)

    # Slightly modify the woc_solution and add them
    while len(woc_population) < population_size:
        modified_solution = copy.deepcopy(woc_solution)  # Deep copy
        mutate(modified_solution)  # Apply mutation
        woc_population.append(modified_solution)

    # Debugging: Check the structure of individuals in woc_population
    for individual in woc_population:
        for job in individual:
            if not isinstance(job, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in job):
                raise ValueError("Individual structure in WoC population is incorrect after initialization.")

    return woc_population
# Function to create the WoC solution
def create_woc_solution(best_solutions):
    job_sequence_votes = {}
    for solution in best_solutions:
        for position, job in enumerate(solution):
            job_tuple = tuple(job)
            if job_tuple not in job_sequence_votes:
                job_sequence_votes[job_tuple] = 0
            job_sequence_votes[job_tuple] += 1

    woc_solution = []
    added_jobs = set()

    # Add most common jobs first
    while len(woc_solution) < len(best_solutions[0]):
        most_common_job_tuple = max(job_sequence_votes, key=lambda k: (job_sequence_votes[k], k not in added_jobs))
        if most_common_job_tuple not in added_jobs:
            woc_solution.append(list(most_common_job_tuple))
            added_jobs.add(most_common_job_tuple)

        # Remove the job from voting to prevent duplicates
        del job_sequence_votes[most_common_job_tuple]

    # If any jobs are missing, add them
    all_jobs = {tuple(job) for solution in best_solutions for job in solution}
    missing_jobs = all_jobs - added_jobs
    for job in missing_jobs:
        woc_solution.append(list(job))

    return woc_solution
# Function to run the program
if __name__ == '__main__':
    jobs = read_jobs_from_file(filename)
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()