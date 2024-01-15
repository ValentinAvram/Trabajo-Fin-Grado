import random
import numpy as np
import time

def imperialist_competitive_algorithm(tsp_instance, number_cities, max_times):
    # Parámetros del algoritmo
    num_countries = 10  # Número de países (soluciones candidatas)
    imperialists_ratio = 0.2  # Proporción de imperialistas con respecto al total de países
    evals = 0
    # Inicialización aleatoria de países
    countries = []
    for _ in range(num_countries):
        country = list(range(0, number_cities))
        random.shuffle(country)
        countries.append(country)

    current_costs = []  # Lista para almacenar los costos de las soluciones
    best_solution = None
    best_cost = float('inf')
    start_time = time.time()

    while time.time() - start_time < max_times:
        # Calcular el costo de cada país
        costs = [tsp_instance.cost(country) for country in countries]

        # Actualizar la mejor solución y su costo
        best_idx = np.argmin(costs)
        if costs[best_idx] < best_cost:
            best_solution = countries[best_idx]
            best_cost = costs[best_idx]

        current_costs.append(best_cost)  # Agregar el mejor costo actual a la lista

        # Seleccionar a los países imperialistas (los mejores)
        num_imperialists = int(num_countries * imperialists_ratio)
        imperialists_idx = np.argsort(costs)[:num_imperialists]

        # Actualizar países colonizados (los peores)
        colonized_idx = np.argsort(costs)[num_imperialists:]

        # Imperialistas compiten y colonizan
        for i in colonized_idx:
            # Selección aleatoria de un imperialista
            imperialist_idx = random.choice(imperialists_idx)

            # Copiar el imperialista y realizar una mutación
            new_country = countries[imperialist_idx][:]
            j, k = random.sample(range(number_cities), 2)
            new_country[j], new_country[k] = new_country[k], new_country[j]

            # Reemplazar el país colonizado si es mejor
            if tsp_instance.cost(new_country) < tsp_instance.cost(countries[i]):
                countries[i] = new_country

        tsp_instance.evaluate_and_resgister(best_solution, 'ICA')
        evals += 1
    return current_costs, best_solution, evals
