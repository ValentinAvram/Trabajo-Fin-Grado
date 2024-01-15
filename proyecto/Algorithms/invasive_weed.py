import random
import time
from Algorithms.common_function_algorithms import read_json

def calculate_suitability(costs):
    # Calculate suitability values based on inverse of costs
    return [1 / cost for cost in costs]

def distribute_seeds(population, suitability):
    total_suitability = sum(suitability)
    num_seeds = [int((s / total_suitability) * len(population)) for s in suitability]
    seeds = []

    for i, num in enumerate(num_seeds):
        seeds.extend([population[i].copy()] * num)
    return seeds


def invasive_weed_optimization(tsp_instance, number_cities, max_time):
    # Initialize the population with random permutations
    hyperparameters = read_json()
    initial_population_size = hyperparameters['initial_population_size_IWO']
    population = [list(range(number_cities)) for _ in range(initial_population_size)]
    evals = 0
    for i in range(initial_population_size):
        random.shuffle(population[i])

    current_costs = []
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Evaluate the population
        costs = [tsp_instance.cost(solution) for solution in population]

        # Calculate suitability based on the costs
        suitability = calculate_suitability(costs)

        # Distribute seeds based on suitability
        seeds = distribute_seeds(population, suitability)

        # Propagation: Generate new solutions by shuffling seeds
        new_population = [random.sample(seed, number_cities) for seed in seeds]

        # Merge the old population and new population
        population.extend(new_population)

        # Select the best solutions from the merged population
        population.sort(key=lambda x: tsp_instance.cost(x))
        population = population[:initial_population_size]

        # Append the current best cost to the list
        current_costs.append(tsp_instance.cost(population[0]))
        tsp_instance.evaluate_and_resgister(population[0], 'IWO')
        evals += 1
    return current_costs, population[0], evals
