import numpy as np
import random
import time
from Algorithms.common_function_algorithms import read_json
def clone(antibody, num_clones):
    clones = [antibody.copy() for _ in range(num_clones)]
    return clones

def mutate(antibody):
    idx1, idx2 = random.sample(range(len(antibody)), 2)
    antibody[idx1], antibody[idx2] = antibody[idx2], antibody[idx1]
    return antibody

def artificial_immune_system(tsp_inst, number_cities, max_time):
    hyperparameters = read_json()
    num_clones = hyperparameters["num_clones"]
    mutation_rate = hyperparameters["mutation_rate_AIS"]
    evals = 0
    antibodies = [list(np.random.permutation(number_cities)) for _ in range(num_clones)]
    current_costs = []

    start_time = time.time()
    while time.time() - start_time < max_time:
        # Clone and mutate antibodies
        clones = []
        for antibody in antibodies:
            num_clones = random.randint(1, 5)  # Random number of clones
            clones.extend(clone(antibody, num_clones))

        # Mutate clones
        for clone_idx in range(len(clones)):
            if random.random() < mutation_rate:
                clones[clone_idx] = mutate(clones[clone_idx])

        # Select antibodies based on cost
        antibodies = sorted(clones, key=lambda x: tsp_inst.cost(x))[:num_clones]

        tsp_inst.evaluate_and_resgister(antibodies[0], 'AIS')
        evals += 1

        # Record the cost of the best antibody
        current_costs.append(tsp_inst.cost(antibodies[0]))

    return current_costs, antibodies[0], evals