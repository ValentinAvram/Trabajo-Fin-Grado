import numpy as np
import random
import time

def two_opt(perm, idx1, idx2):
    new_perm = perm.copy()
    new_perm[idx1:idx2] = new_perm[idx1:idx2][::-1]
    return new_perm

def two_swap(perm, idx1, idx2):
    new_perm = perm.copy()
    new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
    return new_perm

def evaluate_population(population, tsp_instance):
    costs = []
    for perm in population:
        cost = tsp_instance.cost(perm)
        costs.append(cost)
    return costs

def genetic_algorithm(tsp_instance, number_cities, max_time):
    population_size = 50
    mutation_rate = 0.2
    tournament_size = 5
    evals = 0
    current_costs = []
    best_solution = None
    best_cost = float('inf')

    # Initialize population with random permutations
    population = [np.random.permutation(number_cities) for _ in range(population_size)]

    start_time = time.time()
    while time.time() - start_time < max_time:
        # Evaluate the population
        costs = evaluate_population(population, tsp_instance)

        # Find the best solution in the current population
        min_cost_index = np.argmin(costs)
        if costs[min_cost_index] < best_cost:
            best_solution = population[min_cost_index]
            best_cost = costs[min_cost_index]

        # Create a new population through selection and crossover
        new_population = []

        for _ in range(population_size):
            parent1 = population[random.randint(0, population_size - 1)]
            parent2 = population[random.randint(0, population_size - 1)]

            # Tournament selection
            selected_parent = parent1 if evaluate_population([parent1], tsp_instance)[0] < evaluate_population([parent2], tsp_instance)[0] else parent2

            # Apply crossover (uniform crossover)
            child = [selected_parent[i] for i in range(number_cities)]

            # Apply mutation
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    idx1, idx2 = random.sample(range(number_cities), 2)
                    child = two_opt(child, idx1, idx2)
                else:
                    idx1, idx2 = random.sample(range(number_cities), 2)
                    child = two_swap(child, idx1, idx2)

            new_population.append(np.array(child))

            tsp_instance.evaluate_and_resgister(child, 'GA')
            current_costs.append(tsp_instance.cost(child))
            evals += 1
            if time.time() - start_time > max_time:
                break

        population = new_population

    return current_costs, best_solution, evals