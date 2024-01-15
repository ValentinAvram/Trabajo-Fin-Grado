import numpy as np
import time

def two_opt(tour, i, j):
    new_tour = tour.copy()
    new_tour[i:j + 1] = np.flip(tour[i:j + 1])
    return new_tour

def two_swap(tour, i, j):
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def variable_neighbourhood_search(tsp_instance, number_cities, max_time):
    best_permutation = np.random.permutation(number_cities)
    best_cost = tsp_instance.cost(best_permutation)
    current_costs = []
    start_time = time.time()
    evals = 0
    while time.time() - start_time < max_time:

        current_permutation = best_permutation.copy()
        current_cost = best_cost
        # VND (Variable Neighbourhood Descent)
        k = 1
        while k <= 2:
            if k == 1:
                neighbourhood_operator = two_opt
            else:
                neighbourhood_operator = two_swap

            improved = True
            while improved:
                improved = False
                for i in range(number_cities):
                    for j in range(i + 1, number_cities):
                        new_permutation = neighbourhood_operator(current_permutation, i, j)
                        new_cost = tsp_instance.cost(new_permutation)
                        if new_cost < current_cost:
                            current_permutation = new_permutation
                            current_cost = new_cost
                            improved = True
                            break
                    tsp_instance.evaluate_and_resgister(current_permutation, 'VNS')
                    current_costs.append(current_cost)
                    evals += 1
                    if time.time() - start_time > max_time:
                        break

                k += 1  # Move to the next neighbourhood structure

        if current_cost < best_cost:
            best_permutation = current_permutation.copy()
            best_cost = current_cost

    current_costs.sort()

    return current_costs, best_permutation, evals