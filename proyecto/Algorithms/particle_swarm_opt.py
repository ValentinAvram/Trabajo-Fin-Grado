import time
import numpy as np

from Algorithms.common_function_algorithms import read_json

def calculate_cost(tsp_instance, permutation):
    return tsp_instance.cost(permutation)


def generate_unique_permutation(number_cities):
    permutation = list(np.random.permutation(number_cities))
    return permutation


def particle_swarm_optimization(tsp_instance, number_cities, max_time):
    hyperparameters = read_json()
    num_particles = hyperparameters['num_particles_PSO']
    inertia_weight = hyperparameters['inertia_weight_PSO']
    cognitive_weight = hyperparameters['cognitive_weight_PSO']
    social_weight = hyperparameters['social_weight_PSO']
    max_velocity = hyperparameters['max_velocity_PSO']

    particles = [generate_unique_permutation(number_cities) for _ in range(num_particles)]
    velocities = [np.random.uniform(-max_velocity, max_velocity, number_cities) for _ in range(num_particles)]

    global_best_permutation = None
    global_best_cost = np.inf
    current_costs = []
    evals = 0
    start_time = time.time()

    while time.time() - start_time < max_time:
        for i in range(num_particles):
            current_permutation = particles[i]
            current_cost = calculate_cost(tsp_instance, current_permutation)

            if current_cost < global_best_cost:
                global_best_permutation = current_permutation.copy()
                global_best_cost = current_cost

            cognitive_update = cognitive_weight * np.random.random() * (
                    np.array(particles[i]) - np.array(current_permutation))
            social_update = social_weight * np.random.random() * (
                    np.array(global_best_permutation) - np.array(current_permutation))
            velocities[i] = (inertia_weight * np.array(velocities[i])) + cognitive_update + social_update
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            new_permutation = np.array(current_permutation) + velocities[i]
            new_permutation = np.clip(new_permutation, 0, number_cities - 1)
            new_permutation = new_permutation.astype(int)

            # Eliminar valores duplicados en la nueva permutación
            unique_values, counts = np.unique(new_permutation, return_counts=True)
            duplicate_values = unique_values[counts > 1]

            for duplicate_value in duplicate_values:
                available_values = np.setdiff1d(np.arange(number_cities), new_permutation)
                new_value = np.random.choice(available_values)
                new_permutation[np.where(new_permutation == duplicate_value)[0][0]] = new_value

            particles[i] = new_permutation.tolist()
            tsp_instance.evaluate_and_resgister(particles[i], 'PSO')
            current_costs.append(current_cost)
            evals += 1
            if time.time() - start_time > max_time:
                break

    return current_costs, global_best_permutation, evals
