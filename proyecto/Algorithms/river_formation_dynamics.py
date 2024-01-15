import numpy as np
import time

def two_opt(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j + 1] = np.flip(solution[i:j + 1])
    return new_solution

def river_formation_dynamics(tsp_instance, number_cities, max_time):
    def evaluate_river(river):
        return tsp_instance.cost(river)

    def generate_unique_permutation(number_cities):
        return np.random.permutation(number_cities)

    def generate_new_permutation_with_no_repeats(current_permutation):
        new_permutation = current_permutation.copy()
        unused_values = list(set(range(number_cities)) - set(new_permutation))

        for i in range(len(new_permutation)):
            if np.random.rand() < erosion_rate or len(unused_values) == 0:
                continue

            current_value = new_permutation[i]
            new_value = unused_values.pop()
            new_permutation[i] = new_value
            unused_values.append(current_value)

        return new_permutation

    current_permutation = generate_unique_permutation(number_cities)
    current_cost = evaluate_river(current_permutation)
    current_costs = []
    evals = 0
    erosion_rate = 0.1
    start_time = time.time()

    while time.time() - start_time < max_time:
        idx1, idx2 = np.random.choice(number_cities, 2, replace=False)
        new_permutation = two_opt(current_permutation, idx1, idx2)

        new_permutation = generate_new_permutation_with_no_repeats(new_permutation)

        new_cost = evaluate_river(new_permutation)

        if new_cost < current_cost:
            current_permutation = new_permutation
            current_cost = new_cost

        current_costs.append(current_cost)
        tsp_instance.evaluate_and_resgister(new_permutation, 'RFD')
        evals += 1

    return current_costs, current_permutation, evals
