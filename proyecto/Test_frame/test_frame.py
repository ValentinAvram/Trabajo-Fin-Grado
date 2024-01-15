import time
import pandas as pd
import matplotlib.pyplot as plt
import os

from collections import Counter
from TSP_Instance_Experiment.tsp_experiment import TSP_Experiment_Generator

from Algorithms.random_search import random_search
from Algorithms.first_local_search import first_local_search
from Algorithms.best_local_search import best_local_search
from Algorithms.tabu_search import tabu_search
from Algorithms.simulated_annealing import simulated_annealing
from Algorithms.ant_colony_opt import ant_colony_optimization
from Algorithms.bee_colony_opt import bee_colony_optimization
from Algorithms.bat_algorithm import bat_algorithm
from Algorithms.particle_swarm_opt import particle_swarm_optimization
from Algorithms.gravitational_search import gravitational_search
from Algorithms.black_hole_algorithm import black_hole_algorithm
from Algorithms.scatter_search import scatter_search
from Algorithms.cultural_algorithm import cultural_algorithm
from Algorithms.variable_neighbourhood_search import variable_neighbourhood_search
from Algorithms.harmony_search import harmony_search
from Algorithms.imperialist_competitive import imperialist_competitive_algorithm
from Algorithms.chaos_algorithm import chaos_algorithm_optimization
from Algorithms.artificial_inmune_system import artificial_immune_system
from Algorithms.river_formation_dynamics import river_formation_dynamics
from Algorithms.firefly_algorithm import firefly_algorithm
from Algorithms.genetic_algorithm import genetic_algorithm
from Algorithms.crow_search import crow_search_algorithm
from Algorithms.guided_local_search import guided_local_search
from Algorithms.invasive_weed import invasive_weed_optimization
from Algorithms.differential_evolution import differential_evolution

from Test_frame.numerical_test import wilcoxon_test, friedmann_test, nemenyi_test

def test_frame(number_problems, max_evals, min_cities, max_cities):
    experiment = TSP_Experiment_Generator(number_problems, min_cities, max_cities, max_runtime=200)

    plt.rcParams['figure.figsize'] = (19.20, 10.80)
    names_list = []
    best_costs = []

    for i in range(number_problems):
        data_list = []
        permutation_list = []

        N = experiment.set_of_cities[i].N
        set_cities = experiment.set_of_cities[i]

        start_time = time.time()
        rs_costs, best_rs_perm, num_evals = random_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["RS", min(rs_costs), max(rs_costs), (sum(rs_costs) / len(rs_costs)), end_time, num_evals,best_rs_perm])
        best_costs.append(('RS', min(rs_costs)))
        permutation_list.append(('RS', best_rs_perm))
        print('Problem: ', i, ' RS Done')

        start_time = time.time()
        fls_costs, best_fls_perm, num_evals = first_local_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["FLS", min(fls_costs), max(fls_costs), (sum(fls_costs) / len(fls_costs)), end_time, num_evals,
             best_fls_perm])
        best_costs.append(('FLS', min(fls_costs)))
        permutation_list.append(('FLS', best_fls_perm))
        print('Problem: ', i, ' FLS Done')

        start_time = time.time()
        bls_costs, best_bls_perm, num_evals = best_local_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["BLS", min(bls_costs), max(bls_costs), (sum(bls_costs) / len(bls_costs)), end_time, num_evals,
             best_bls_perm])
        best_costs.append(('BLS', min(bls_costs)))
        permutation_list.append(('BLS', best_bls_perm))
        print('Problem: ', i, ' BLS Done')

        start_time = time.time()
        tabu_costs, best_tabu_perm, num_evals = tabu_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["TS", min(tabu_costs), max(tabu_costs), (sum(tabu_costs) / len(tabu_costs)), end_time, num_evals,
             best_tabu_perm])
        best_costs.append(('TS', min(tabu_costs)))
        permutation_list.append(('TS', best_tabu_perm))
        print('Problem: ', i, ' TS Done')

        start_time = time.time()
        sim_ann_costs, best_sim_ann_perm, num_evals = simulated_annealing(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["SimAn", min(sim_ann_costs), max(sim_ann_costs), (sum(sim_ann_costs) / len(sim_ann_costs)),
             end_time, num_evals, best_sim_ann_perm])
        best_costs.append(('SimAn', min(sim_ann_costs)))
        permutation_list.append(('SimAn', best_sim_ann_perm))
        print('Problem: ', i, ' SimAn Done')

        start_time = time.time()
        aco_costs, best_aco_perm, num_evals = ant_colony_optimization(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["ACO", min(aco_costs), max(aco_costs), (sum(aco_costs) / len(aco_costs)), end_time, num_evals,
             best_aco_perm])
        best_costs.append(('ACO', min(aco_costs)))
        permutation_list.append(('ACO', best_aco_perm))
        print('Problem: ', i, ' ACO Done')

        start_time = time.time()
        bco_costs, best_bco_perm, num_evals = bee_colony_optimization(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["BCO", min(bco_costs), max(bco_costs), (sum(bco_costs) / len(bco_costs)), end_time, num_evals,
             best_bco_perm])
        best_costs.append(('BCO', min(bco_costs)))
        permutation_list.append(('BCO', best_bco_perm))
        print('Problem: ', i, ' BCO Done')

        start_time = time.time()
        pso_costs, best_pso_perm, num_evals = particle_swarm_optimization(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["PSO", min(pso_costs), max(pso_costs), (sum(pso_costs) / len(pso_costs)), end_time, num_evals,
             best_pso_perm])
        best_costs.append(('PSO', min(pso_costs)))
        permutation_list.append(('PSO', best_pso_perm))
        print('Problem: ', i, ' PSO Done')

        start_time = time.time()
        gs_costs, best_gs_perm, num_evals = gravitational_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["GS", min(gs_costs), max(gs_costs), (sum(gs_costs) / len(gs_costs)), end_time, num_evals,
             best_gs_perm])
        best_costs.append(('GS', min(gs_costs)))
        permutation_list.append(('GS', best_gs_perm))
        print('Problem: ', i, ' GS Done')

        start_time = time.time()
        bh_costs, best_bh_perm, num_evals = black_hole_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["BHA", min(bh_costs), max(bh_costs), (sum(bh_costs) / len(bh_costs)), end_time, num_evals,
             best_bh_perm])
        best_costs.append(('BHA', min(bh_costs)))
        permutation_list.append(('BHA', best_bh_perm))
        print('Problem: ', i, ' BHA Done')

        start_time = time.time()
        ss_costs, best_ss_perm, num_evals = scatter_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["SS", min(ss_costs), max(ss_costs), (sum(ss_costs) / len(ss_costs)), end_time, num_evals, best_ss_perm])
        best_costs.append(('SS', min(ss_costs)))
        permutation_list.append(('SS', best_ss_perm))
        print('Problem: ', i, ' SS Done')

        start_time = time.time()
        ca_costs, best_ca_perm, num_evals = cultural_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(["CA", min(ca_costs), max(ca_costs), (sum(ca_costs) / len(ca_costs)), end_time, num_evals,
                          best_ca_perm])
        best_costs.append(('CA', min(ca_costs)))
        permutation_list.append(('CA', best_ca_perm))
        print('Problem: ', i, ' CA Done')

        start_time = time.time()
        vns_costs, best_vns_perm, num_evals = variable_neighbourhood_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["VNS", min(vns_costs), max(vns_costs), (sum(vns_costs) / len(vns_costs)),
             end_time, num_evals, best_vns_perm])
        best_costs.append(('VNS', min(vns_costs)))
        permutation_list.append(('VNS', best_vns_perm))
        print('Problem: ', i, ' VNS Done')

        start_time = time.time()
        hs_costs, best_hs_perm, num_evals = harmony_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["HS", min(hs_costs), max(hs_costs), (sum(hs_costs) / len(hs_costs)), end_time, num_evals, best_hs_perm])
        best_costs.append(('HS', min(hs_costs)))
        permutation_list.append(('HS', best_hs_perm))
        print('Problem: ', i, ' HS Done')

        start_time = time.time()
        ica_costs, best_ica_perm, num_evals = imperialist_competitive_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["ICA", min(ica_costs), max(ica_costs), (sum(ica_costs) / len(ica_costs)),
             end_time, num_evals, best_ica_perm])
        best_costs.append(('ICA', min(ica_costs)))
        permutation_list.append(('ICA', best_ica_perm))
        print('Problem: ', i, ' ICA Done')

        start_time = time.time()
        chaos_costs, best_chaos_perm, num_evals = chaos_algorithm_optimization(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["CAO", min(chaos_costs), max(chaos_costs), (sum(chaos_costs) / len(chaos_costs)),
             end_time, num_evals, best_chaos_perm])
        best_costs.append(('CAO', min(chaos_costs)))
        permutation_list.append(('CAO', best_chaos_perm))
        print('Problem: ', i, ' CAO Done')

        start_time = time.time()
        ais_costs, best_ais_perm, num_evals = artificial_immune_system(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["AIS", min(ais_costs), max(ais_costs), (sum(ais_costs) / len(ais_costs)), end_time, num_evals,
             best_ais_perm])
        best_costs.append(('AIS', min(ais_costs)))
        permutation_list.append(('AIS', best_ais_perm))
        print('Problem: ', i, ' AIS Done')

        start_time = time.time()
        rfd_costs, best_rfd_perm, num_evals = river_formation_dynamics(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["RFD", min(rfd_costs), max(rfd_costs), (sum(rfd_costs) / len(rfd_costs)), end_time, num_evals,
             best_rfd_perm])
        best_costs.append(('RFD', min(rfd_costs)))
        permutation_list.append(('RFD', best_rfd_perm))
        print('Problem: ', i, ' RFD Done')

        start_time = time.time()
        fa_costs, best_fa_perm, num_evals = firefly_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(["FA", min(fa_costs), max(fa_costs), (sum(fa_costs) / len(fa_costs)), end_time, num_evals,
                          best_fa_perm])
        best_costs.append(('FA', min(fa_costs)))
        permutation_list.append(('FA', best_fa_perm))
        print('Problem: ', i, ' FA Done')

        start_time = time.time()
        ga_costs, best_ga_perm, num_evals = genetic_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(["GA", min(ga_costs), max(ga_costs), (sum(ga_costs) / len(ga_costs)), end_time, num_evals,
                          best_ga_perm])
        best_costs.append(('GA', min(ga_costs)))
        permutation_list.append(('GA', best_ga_perm))
        print('Problem: ', i, ' GA Done')

        start_time = time.time()
        gls_costs, best_gls_perm, num_evals = guided_local_search(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["GLS", min(gls_costs), max(gls_costs), (sum(gls_costs) / len(gls_costs)), end_time, num_evals,
             best_gls_perm])
        best_costs.append(('GLS', min(gls_costs)))
        permutation_list.append(('GLS', best_gls_perm))
        print('Problem: ', i, ' GLS Done')

        start_time = time.time()
        csa_costs, best_csa_perm, num_evals = crow_search_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["CSA", min(csa_costs), max(csa_costs), (sum(csa_costs) / len(csa_costs)), end_time, num_evals,
             best_csa_perm])
        best_costs.append(('CSA', min(csa_costs)))
        permutation_list.append(('CSA', best_csa_perm))
        print('Problem: ', i, ' CSA Done')

        start_time = time.time()
        iwo_costs, best_iw_perm, num_evals = invasive_weed_optimization(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["IWO", min(iwo_costs), max(iwo_costs), (sum(iwo_costs) / len(iwo_costs)), end_time, num_evals,
             best_iw_perm])
        best_costs.append(('IWO', min(iwo_costs)))
        permutation_list.append(('IWO', best_iw_perm))
        print('Problem: ', i, ' IWO Done')

        start_time = time.time()
        ba_costs, best_ba_perm, num_evals = bat_algorithm(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["BA", min(ba_costs), max(ba_costs), (sum(ba_costs) / len(ba_costs)), end_time, num_evals, best_ba_perm])
        best_costs.append(('BA', min(ba_costs)))
        permutation_list.append(('BA', best_ba_perm))
        print('Problem: ', i, ' BA Done')

        start_time = time.time()
        de_costs, best_de_perm, num_evals = differential_evolution(set_cities, N, max_evals)
        end_time = time.time() - start_time
        data_list.append(
            ["DE", min(de_costs), max(de_costs), (sum(de_costs) / len(de_costs)), end_time, num_evals,
             best_de_perm])
        best_costs.append(('DE', min(de_costs)))
        permutation_list.append(('DE', best_de_perm))
        print('Problem: ', i, ' DE Done')

        plot_best_perm(set_cities, permutation_list, i)

        # NOTE: Already done MOGA and CuckooSearch, but not included in the experiment
        df_results = pd.DataFrame(data_list, columns=["Algorithm", "Best Cost", "Worst Cost", "Average Cost", "Runtime", "Evaluations",
                                                      "Best permutation"])
        df_results['Cost Rank'] = df_results['Best Cost'].rank(method='min')
        df_results['Runtime Rank'] = df_results['Runtime'].rank(method='min')
        df_results['Evaluations Rank'] = df_results['Evaluations'].rank(method='min')

        name = "data/results" + str(i) + ".csv"
        df_results.to_csv(name, index=False)

        df_results = df_results.sort_values(by=['Cost Rank'])

        list = df_results['Algorithm'].head(25).tolist()
        names_list.append(list)

    best_algs = [valor for sublista in names_list for valor in sublista]
    checker = Counter(best_algs)

    top_worst = [valor for valor, frecuencia in checker.most_common()[:-6:-1]]

    top_2 = checker.most_common(2)
    top_2 = [tupla[0] for tupla in top_2]

    top_3 = checker.most_common(3)
    top_3 = [tupla[0] for tupla in top_3]

    top_5 = checker.most_common(5)
    top_5 = [tupla[0] for tupla in top_5]

    top_10 = checker.most_common(10)
    top_10 = [tupla[0] for tupla in top_10]

    hybrid_methods = ['CA', 'VNS', 'GLS']
    evolutionary_methods = ['GA', 'DE', 'ICA', 'AIS']
    ls_methods = ['FLS', 'BLS', 'TS', 'GLS']
    swarm_methods = ['ACO', 'BCO', 'PSO', 'CSA', 'HS']
    population_methods = ['PSO', 'GA', 'CA', 'ICA', 'AIS', 'ACO', 'BCO', 'RFD', 'BA', 'IWO']
    nature_methods = ['ACO', 'BCO', 'FA', 'RFD', 'BA', 'IWO']

    csv_files = os.listdir('data')
    df_combined = pd.DataFrame()
    for file in csv_files:
        if file.startswith('results'):
            df = pd.read_csv('data/' + file)
            df = df.sort_values(by=['Cost Rank'])
            df_combined = pd.concat([df_combined, df])

    df_combined = df_combined.groupby('Algorithm').agg({'Cost Rank': 'mean', 'Runtime': 'mean'}).reset_index()
    df_combined = df_combined.sort_values(by=['Cost Rank'])

    df_combined.to_csv('data/results.csv', index=False)

    ####
    res1 = wilcoxon(best_costs, top_2)
    res2 = wilcoxon(best_costs, top_worst)

    res3 = friedmann(best_costs, top_worst, '5 Peores')
    res6 = friedmann(best_costs, top_3, '3 Mejores')
    res7 = friedmann(best_costs, top_5, '5 Mejores')
    res8 = friedmann(best_costs, top_10, '10 Mejores')
    res9 = friedmann(best_costs, hybrid_methods, 'Híbridos')
    res10 = friedmann(best_costs, evolutionary_methods, 'Evolutivos')
    res11 = friedmann(best_costs, ls_methods, 'Búsqueda Local')
    res12 = friedmann(best_costs, swarm_methods, 'Enjambre')
    res13 = friedmann(best_costs, population_methods, 'Poblacionales')
    res14 = friedmann(best_costs, nature_methods, 'Naturales')
    res15 = friedmann(best_costs, list, 'Todos')


    if os.path.exists('data/best_costs.csv'):
        os.remove('data/best_costs.csv')

    df_bests = pd.DataFrame(best_costs, columns=['Alg', 'Best Cost'])
    df_bests.to_csv('data/best_costs.csv', index=False)

    if os.path.exists('data/test_results.txt'):
        os.remove('data/test_results.txt')

    with open('data/tests_results.txt', 'a') as archivo:
        archivo.write('Mejores: ' + res1)
        archivo.write('Peores: ' + res2 + '\n')
        archivo.write(res3)
        archivo.write(res6)
        archivo.write(res7)
        archivo.write(res8)
        archivo.write(res9)
        archivo.write(res10)
        archivo.write(res11)
        archivo.write(res12)
        archivo.write(res13)
        archivo.write(res14)
        archivo.write(res15)

        # archivo.write('\nNemenyi 5 Mejores: CD:' + nemenyi1)
        # archivo.write('\nNemenyi 10 Mejores: CD:' + nemenyi2)
        # archivo.write('\nNemenyi 5 Peores: CD:' + nemenyi3)
        # archivo.write('\nNemenyi Híbridos: CD:' + nemenyi5)
        # archivo.write('\nNemenyi Evolutivos: CD:' + nemenyi6)
        # archivo.write('\nNemenyi Búsqueda Local: CD:' + nemenyi7)
        # archivo.write('\nNemenyi Enjambre: CD:' + nemenyi8)
        # archivo.write('\nNemenyi Poblacionales: CD:' + nemenyi9)
        # archivo.write('\nNemenyi Naturales: CD:' + nemenyi10)

    plot_ranking(experiment)
    plot_ranking_topN(experiment, top_3, 3)
    plot_ranking_topN(experiment, top_5, 5)
    plot_ranking_topN(experiment, top_10, 10)
    plot_ranking_topN(experiment, top_worst, 'worst')
    plot_ranking_topN(experiment, hybrid_methods, 'hybrid')
    plot_ranking_topN(experiment, evolutionary_methods, 'evolutionary')
    plot_ranking_topN(experiment, ls_methods, 'ls')
    plot_ranking_topN(experiment, swarm_methods, 'swarm')
    plot_ranking_topN(experiment, population_methods, 'population')
    plot_ranking_topN(experiment, nature_methods, 'nature')

    plot_convergence(experiment)
    plot_convergenceN(experiment, top_3, 3)
    plot_convergenceN(experiment, top_5, 5)
    plot_convergenceN(experiment, top_10, 10)
    plot_convergenceN(experiment, top_worst, 'worst')
    plot_convergenceN(experiment, hybrid_methods, 'hybrid')
    plot_convergenceN(experiment, evolutionary_methods, 'evolutionary')
    plot_convergenceN(experiment, ls_methods, 'ls')
    plot_convergenceN(experiment, swarm_methods, 'swarm')
    plot_convergenceN(experiment, population_methods, 'population')
    plot_convergenceN(experiment, nature_methods, 'nature')

    return None


def wilcoxon(best_costs, names_list):

    values_alg1 = [lista for nombre, lista in best_costs if nombre == names_list[0]]
    values_alg2 = [lista for nombre, lista in best_costs if nombre == names_list[1]]

    type = f'{names_list[0]} vs {names_list[1]}'
    var = wilcoxon_test(values_alg1, values_alg2, type)

    return var

def friedmann(best_costs, names_list, tipo):
    values = []

    for i in range(len(names_list)):
        indv_res = [lista for nombre, lista in best_costs if nombre == names_list[i]]
        values.append(indv_res)

    text = friedmann_test(values, names_list, tipo)

    return text

def nemenyi(best_costs, names_list, tipo):
    values = []

    for i in range(len(names_list)):
        indv_res = [valor for nombre, valor in best_costs if nombre == names_list[i]]
        alg_res = []
        alg_res.extend(indv_res)

        values.append(alg_res)

    cd_value = nemenyi_test(values, names_list, tipo)

    return  cd_value

def plot_convergence(experiment):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    experiment.plot_convergence_graphs(ax)

    # save the figure
    name = 'data/convergence_graph.png'
    plt.savefig(name)
    plt.close()

def plot_convergenceN(experiment, best_algs, n):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    experiment.plot_convergence_N(ax, best_algs)

    # save the figure
    name = f'data/convergence_{n}.png'
    plt.savefig(name)
    plt.close()

def plot_ranking(experiment):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    experiment.plot_rank_evolution_graph(ax)

    name = 'data/rank_evolution_graph.png'
    plt.savefig(name)
    plt.close()

def plot_ranking_topN(experiment, top_N, n):
    fig, ax = plt.subplots()
    ax.set_xscale('log')

    experiment.plot_rank_topN(ax, top_N)
    name = f'data/rank_{n}.png'
    plt.savefig(name)
    plt.close()


def plot_best_perm(tsp_instance, permutations_list, iteration):
    for i in range(len(permutations_list)):
        fig, ax = plt.subplots(ncols=2)
        tsp_instance.plot_cities(ax[0])
        tsp_instance.draw_circuit(permutations_list[i][1], ax[1], 'r--', 1)

        alg_name = permutations_list[i][0]
        print('Plotting ' + alg_name + ' best permutation')
        name = 'data/problem_' + str(iteration) + '_' + alg_name + '.png'
        plt.savefig(name)
        plt.close()