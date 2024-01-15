import numpy as np
import time

def two_opt(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j + 1] = np.flip(solution[i:j + 1])
    return new_solution

def two_swap(solution, i, j):
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def calculate_force(m1, m2, r):
    G = 6.67430e-11  # Gravitational constant
    return (G * m1 * m2) / (r ** 2)

def update_agent(agent, force):
    acceleration = force / agent['mass']
    agent['velocity'] += acceleration
    agent['position'] += agent['velocity'].astype(int)  # Convert velocity to integer before updating position

def gravitational_search(tsp_instance, number_cities, max_time):
    G = 20  # Number of agents (solutions)
    dim = number_cities  # Number of cities
    lb = 0  # Lower bound for city indices
    ub = dim - 1  # Upper bound for city indices
    evals = 0
    agents = []
    for _ in range(G):
        position = np.random.permutation(dim)
        mass = 1.0
        velocity = np.zeros(dim)
        agents.append({'position': position, 'mass': mass, 'velocity': velocity})

    best_solution = None
    best_cost = float('inf')
    current_costs = []

    start_time = time.time()
    while time.time() - start_time < max_time:
        for i in range(G):
            current_solution = agents[i]['position']
            current_cost = tsp_instance.cost(current_solution)

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()

            for j in range(G):
                if i != j:
                    distance = np.linalg.norm(agents[i]['position'] - agents[j]['position'])
                    force = calculate_force(agents[i]['mass'], agents[j]['mass'], distance)
                    update_agent(agents[i], force)

            tsp_instance.evaluate_and_resgister(current_solution, 'GS')
            current_costs.append(current_cost)
            evals += 1
            if time.time() - start_time >= max_time:
                break

        # Apply the two operators of neighborhoods
        for i in range(G):
            r = np.random.random()
            if r < 0.5:
                k = np.random.randint(0, dim)
                l = np.random.randint(0, dim)
                agents[i]['position'] = two_opt(agents[i]['position'], k, l)
            else:
                k = np.random.randint(0, dim)
                l = np.random.randint(0, dim)
                agents[i]['position'] = two_swap(agents[i]['position'], k, l)

    return current_costs, best_solution, evals
