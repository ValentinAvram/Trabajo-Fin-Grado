import numpy as np
import time
from Algorithms.common_function_algorithms import generate_permutation, read_json


def ant_colony_optimization(tsp_instance, number_cities, max_time):
    hyperparameters = read_json()
    alpha = hyperparameters["alpha"]
    beta = hyperparameters["beta"]
    rho = hyperparameters["rho"]
    initial_pheromone = hyperparameters["initial_pheromone"]

    pheromone_matrix = initialize_pheromone_matrix(number_cities, initial_pheromone)
    best_permutation = generate_permutation(number_cities)
    best_cost = tsp_instance.cost(best_permutation)
    current_costs = []
    evals = 0

    start_time = time.time()
    while time.time() - start_time < max_time:
        ant_permutations = []
        ant_costs = []

        for ant in range(number_cities):
            current_city = np.random.randint(number_cities)
            ant_path = [current_city]
            unvisited_cities = set(range(number_cities))
            unvisited_cities.remove(current_city)

            while unvisited_cities:
                next_city = np.random.choice(list(unvisited_cities),
                                             p=get_probabilities(tsp_instance, pheromone_matrix, current_city,
                                                                 unvisited_cities,
                                                                 alpha, beta))
                ant_path.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city

            ant_permutations.append(ant_path)
            ant_costs.append(tsp_instance.cost(ant_path))

            current_costs.append(tsp_instance.cost(ant_path))
            tsp_instance.evaluate_and_resgister(ant_path, 'ACO')
            evals += 1
            if time.time() - start_time > max_time:
                break


        # Update pheromone matrix
        update_pheromones(pheromone_matrix, ant_permutations, ant_costs, rho)

        # Find the best ant solution
        best_ant_index = np.argmin(ant_costs)
        if ant_costs[best_ant_index] < best_cost:
            best_permutation = ant_permutations[best_ant_index]

    return current_costs, best_permutation, evals


def get_probabilities(tsp_instance, pheromone_matrix, current_city, unvisited_cities, alpha, beta):
    probabilities = []
    total = 0

    for city in unvisited_cities:
        pheromone = pheromone_matrix[current_city][city]
        distance = 1 / tsp_instance.distances[current_city][city]
        probabilities.append((pheromone ** alpha) * (distance ** beta))
        total += probabilities[-1]

    probabilities = [p / total for p in probabilities]
    return probabilities


def update_pheromones(pheromone_matrix, ant_permutations, ant_costs, rho):
    pheromone_matrix *= (1 - rho)

    for ant_path, cost in zip(ant_permutations, ant_costs):
        for i in range(len(ant_path)):
            pheromone_matrix[ant_path[i]][ant_path[(i + 1) % len(ant_path)]] += 1 / cost


def initialize_pheromone_matrix(number_cities, initial_pheromone):
    return np.full((number_cities, number_cities), initial_pheromone)
