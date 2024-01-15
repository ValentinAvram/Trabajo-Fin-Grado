import numpy as np
import random
import time

def two_opt(solution, idx1, idx2):
    new_solution = solution.copy()
    new_solution[idx1:idx2] = reversed(solution[idx1:idx2])
    return new_solution

def two_swap(solution, idx1, idx2):
    new_solution = solution.copy()
    new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
    return new_solution

def genetic_algorithm_multiobjective(tsp_instance, number_cities, max_time):
    def crossover(parent1, parent2):
        # Simple ordered crossover (OX1)
        start = random.randint(0, number_cities - 1)
        end = random.randint(start + 1, number_cities)
        child = [-1] * number_cities
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child]
        child[:start], child[end:] = remaining[:start], remaining[start:]
        return child

    def mutate(solution):
        if random.random() < 0.5:
            idx1, idx2 = random.sample(range(number_cities), 2)
            return two_opt(solution, idx1, idx2)
        else:
            idx1, idx2 = random.sample(range(number_cities), 2)
            return two_swap(solution, idx1, idx2)

    current_solution = np.random.permutation(number_cities)
    current_costs = []

    start_time = time.time()

    while time.time() - start_time < max_time:
        offspring = []
        for _ in range(2):
            parent1 = random.choice(offspring) if offspring else current_solution
            parent2 = random.choice(offspring) if offspring else current_solution
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                child = mutate(child)
            offspring.append(child)

        all_solutions = offspring + [current_solution]
        all_costs = [tsp_instance.cost(solution) for solution in all_solutions]

        current_solution = offspring[np.argmax(all_costs)]
        current_costs.append(max(all_costs))

    return current_costs, current_solution