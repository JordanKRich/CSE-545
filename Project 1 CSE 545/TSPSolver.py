import math
from itertools import permutations
import time
import matplotlib.pyplot as plt

#Function to read the tsp file and return the coordinates of the cities
def read_tsp_file(filename):
    with open('Random12.tsp', 'r') as file: # open the file in read mode
        lines = file.readlines()
        relevant_lines = lines[7:] # discard the first 7 lines
        coordinates = [] # initialize coordinates list
        for line in relevant_lines: # read the relevant lines
            parts = line.split() # split the line according to whitespaces
            city_number = int(parts[0]) # first part is the city number
            longitude = float(parts[1]) # second part is the longitude
            latitude = float(parts[2]) # third part is the latitude
            coordinates.append((city_number, longitude, latitude)) # add the city and coordinates to the list
    return coordinates


def calculate_distance(coord1, coord2): #function to calculate the distance between two cities
    _, x1, y1 = coord1 # unpack the coordinates of the first city
    _, x2, y2 = coord2 # unpack the coordinates of the second city

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # calculate the Euclidean distance
    return distance


#function to calculate the total distance of the path
def calculate_total_distance(path, coordinates):
    total_distance = 0 # initialize the total distance to zero
    for i in range(len(path) - 1):
        total_distance += calculate_distance(coordinates[path[i]], coordinates[path[i + 1]]) # add the distance between two consecutive cities
    total_distance += calculate_distance(coordinates[path[-1]], coordinates[path[0]])  # add the distance between the last city and the first city
    return total_distance


#function to solve the tsp problem using brute force permutation
def solve_tsp(coordinates):
    number_of_cities = len(coordinates)
    total_permutations=math.factorial(number_of_cities) #calculate the total number of permutations
    interval= total_permutations//20 #calculate the interval to print the progress
    best_distance = float('inf')  # initialize with positive infinity
    best_path = []

    start_time=time.time() #start the timer
    permutation_count=0 #initialize the permutation count to zero

    for perm in permutations(range(number_of_cities)):
        permutation_count+=1
        total_distance = calculate_total_distance(perm, coordinates)
        if total_distance < best_distance:
            best_distance = total_distance
            best_path = perm #update the best path
        if permutation_count % interval==0:
            print("Progress:",permutation_count,"out of", total_permutations,"permutations") #print the progress of permutations
    end_time=time.time()
    elapsed_time=end_time-start_time
    print("Final Answer\nElapsed Time: ",elapsed_time)
    return best_path, best_distance, elapsed_time

def main():
    filename = r'C:\Users\jorda\OneDrive\Desktop\Project 1 CSE 545\Random12.tsp'
    coordinates = read_tsp_file(filename)

    if coordinates is not None:
        solution = solve_tsp(coordinates)
        print("Optimal Tour:",[city + 1 for city in solution[0]]) # add 1 to each city number to get the original city number
    
    
        best_path=solution[0]

        longitude=[coordinates[i][1] for i in best_path] #extract the longitude of the cities in the best path
        latitude=[coordinates[i][2] for i in best_path] #extract the latitude of the cities in the best path

        #Scatter plot of the cities and start point
        plt.scatter(longitude,latitude, color='black', marker='o', s=100, label='Cities')
        plt.scatter(longitude[0],latitude[0], color='red', marker='x', s=100, label='Start')
        
        #plot lines connecting cities and arrows showing the direction of the path(may have to zoom in to see the arrows)
        for i in range(len(best_path)-1):
            
            plt.plot([longitude[i],longitude[i+1]],
                     [latitude[i],latitude[i+1]],
                      color='blue',
                      linewidth=3
                    ) #plotting the lines connecting the cities
            
            plt.arrow(longitude[i],
                       latitude[i],
                       longitude[i + 1] - longitude[i],
                       latitude[i + 1] - latitude[i], 
                       head_width=0.8,
                       head_length=1,
                       facecolor='red',
                       edgecolor='red'
                     ) #plotting the arrows showing the direction of the path
            
        #connecting the last city to first city
        plt.plot([longitude[-1],longitude[0]],[latitude[-1],latitude[0]], color='blue')

        #Set labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('TSP Problem Visulization')
        plt.legend()

        #show the plot
        plt.show()

    else:
        print("No solution found!")


if __name__ == "__main__":
    timeout_seconds=7200
    main()
