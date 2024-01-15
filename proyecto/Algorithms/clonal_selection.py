import random

from Algorithms.common_function_algorithms import generate_permutation

def clone(current_population, num_clones):
    clones = []
    for i in range(len(current_population)):
        for _ in range(num_clones):
            clones.append(current_population[i])
    return clones


def hypermutation(population, clone_percentage):
    num_clones = int(len(population) * clone_percentage)
    clones = clone(population, num_clones)

    for clone_permutation in clones:
        mutation_rate = random.uniform(0.1, 0.5)
        for _ in range(int(mutation_rate * len(clone_permutation))):
            idx1, idx2 = random.sample(range(len(clone_permutation)), 2)
            clone_permutation[idx1], clone_permutation[idx2] = clone_permutation[idx2], clone_permutation[idx1]

    return clones


def select_best(tsp_instance, population, num_select):
    population = sorted(population, key=lambda x: tsp_instance.cost(x))
    return population[:num_select]


def clonal_selection_algorithm(tsp_instance, number_cities, max_evals):
    current_population = [generate_permutation(number_cities) for _ in range(number_cities)]  # Initial population of permutations
    current_costs = []

    for _ in range(max_evals):
        clones = hypermutation(current_population, 0.2)
        current_population.extend(clones)

        current_population = select_best(tsp_instance, current_population, number_cities)

        best_permutation = current_population[0]
        current_costs.append(tsp_instance.cost(best_permutation))

    best_permutation = current_population[0]
    tsp_instance.results['CS'] = current_costs

    return current_costs, best_permutation