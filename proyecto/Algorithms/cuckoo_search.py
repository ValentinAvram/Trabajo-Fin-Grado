import numpy as np

from Algorithms.common_function_algorithms import generate_permutation

def cuckoo_search(tsp_instance, number_cities, max_evals):
    def generate_random_permutation(number_cities):
        return list(np.random.permutation(number_cities))

    def levy_flight():
        beta = 1.5
        sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        step = u / (abs(v) ** (1 / beta))

        return step

    def get_best_neighbor(current_permutation, current_cost):
        num_cuckoos = 10
        neighbors = [current_permutation.copy() for _ in range(num_cuckoos)]
        neighbor_costs = [current_cost] * num_cuckoos

        for i in range(num_cuckoos):
            step = levy_flight()
            index1, index2 = random.sample(range(number_cities), 2)
            neighbors[i][index1], neighbors[i][index2] = neighbors[i][index2], neighbors[i][index1]
            neighbor_costs[i] = tsp_instance.cost(neighbors[i])

        best_neighbor = neighbors[np.argmin(neighbor_costs)]
        best_neighbor_cost = min(neighbor_costs)

        return best_neighbor, best_neighbor_cost

    current_permutation = generate_random_permutation(number_cities)
    current_cost = tsp_instance.cost(current_permutation)
    current_costs = [current_cost]

    for _ in range(max_evals):
        new_permutation, new_cost = get_best_neighbor(current_permutation, current_cost)

        if new_cost < current_cost:
            current_permutation, current_cost = new_permutation, new_cost

        current_costs.append(current_cost)

    return current_costs, current_permutation