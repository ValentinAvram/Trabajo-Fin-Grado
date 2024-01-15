import numpy as np
import pandas as pd
import scipy.stats as st

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from TSP_Instance_Experiment.tsp_instance import TSP_Instance


class TSP_Experiment_Generator:

    def __init__(self, num_problems, min_num_cities, max_num_cities, max_runtime):
        self.min_num_cities = min_num_cities
        self.max_num_cities = max_num_cities
        self.set_of_cities = []
        self.max_runtime = max_runtime
        self.initialise_problems(num_problems)

    def __str__(self):
        output = '------------TSP GENERATOR BEGIN---\n'
        output += str(len(self.set_of_cities)) + " " + str(self.min_num_cities) + " " + str(self.max_num_cities) + " " + str(self.max_runtime)
        for i_problem in self.set_of_cities:
            output += '\n' + str(i_problem) + '\n'
        output += '------------TSP GENERATOR END----\n'
        return output

    def introduce_a_new_instance(self):
        self.set_of_cities.append(
            TSP_Instance(
                np.random.choice(np.arange(self.min_num_cities,
                                           self.max_num_cities)), self.max_runtime))
    def initialise_problems(self, num_problems):
        for i in range(num_problems):
            self.introduce_a_new_instance()

    def plot_convergence_graphs(self, axes_or_filename, xlogscale=False, ylogscale=False):

        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        if xlogscale:
            ax.set_xscale('log')

        if ylogscale:
            ax.set_yscale('log')

        # Normalize the results of the algorithms on every problem
        normalised_results = []
        for i_problem in self.set_of_cities:
            normalised_results.append(pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()])))
            normalised_results[-1].ffill(axis=0, inplace=True)
            max_value = normalised_results[-1].max().max()
            min_value = normalised_results[-1].min().min()
            normalised_results[-1] -= min_value
            if max_value > min_value:
                normalised_results[-1] /= (max_value - min_value)

            if ylogscale:
                normalised_results[-1] += 0.001

        # Executed algs
        algs = list(set([i for j in self.set_of_cities for i in j.results]))

        for i in algs:
            # Results of algorithm i on all the problems
            df = pd.DataFrame(dict([(index, j[i]) for index, j in enumerate(normalised_results) if i in j.columns]))
            df.ffill(axis=0, inplace=True)
            try:
                ax.step(df.index, np.mean(df, axis=1), where='post', label=i)
            except ValueError:
                print(normalised_results[0].index)
                print(normalised_results[1].index)
                print(normalised_results[2].index)
                print(list(df.index))
                print(df.loc[:20])
                raise
            np.seterr(all='ignore')
            conf_interval = st.norm.interval(confidence=0.95, loc=np.mean(df, axis=1), scale=st.sem(df,axis=1))
            np.seterr(all='raise')
            # ax.fill_between(df.index, np.min(df, axis=1), np.max(df, axis=1), alpha=0.2)
            ax.fill_between(df.index, conf_interval[0], conf_interval[1], alpha=0.2)

        ax.legend()

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def plot_rank_evolution_graph(self, axes_or_filename, xlogscale=False, ylogscale=False):

        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        if xlogscale:
            ax.set_xscale('log')

        if ylogscale:
            ax.set_yscale('log')

        results_ranks = []

        # Executed algs
        algs = list(set([i for j in self.set_of_cities for i in j.results]))

        ax.set_ylim(1,len(algs))

        for i_problem in self.set_of_cities:
            data_results = pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()]))

            for i_alg in algs:
                if i_alg not in data_results.columns:
                    data_results = pd.concat((data_results, pd.DataFrame({i_alg: dict([(1,i_problem.size()*TSP_Instance.sqrt_2)])})), axis=1)

            data_results.ffill(axis=0, inplace=True)
            data_results = data_results.round(decimals=6)
            results_ranks.append(data_results.rank(axis=1))


        for i in algs:
            df = pd.DataFrame({str(index): j[i] for index, j in enumerate(results_ranks)})
            df.ffill(axis=0, inplace=True)
            mean = np.mean(df, axis=1)
            std = np.std(df, axis=1)
            np.seterr(all='ignore')
            conf_interval = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(df,axis=1))
            np.seterr(all='raise')
            ax.step(df.index, mean, where='post', label=i)
            # ax.fill_between(df.index, mean - std, mean + std, alpha=0.2)
            ax.fill_between(df.index, conf_interval[0], conf_interval[1], alpha=0.2)

        ax.legend()

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def clear_results(self):
        for i in self.set_of_cities:
            i.clear_results()

    def plot_convergence_N(self, axes_or_filename, best_algs, xlogscale=False, ylogscale=False):

        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        if xlogscale:
            ax.set_xscale('log')

        if ylogscale:
            ax.set_yscale('log')

        # Normalize the results of the algorithms on every problem
        normalised_results = []
        for i_problem in self.set_of_cities:
            normalised_results.append(pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()])))
            normalised_results[-1].ffill(axis=0, inplace=True)
            max_value = normalised_results[-1].max().max()
            min_value = normalised_results[-1].min().min()
            normalised_results[-1] -= min_value
            if max_value > min_value:
                normalised_results[-1] /= (max_value - min_value)

            if ylogscale:
                normalised_results[-1] += 0.001

        # Executed algs
        algs = list(set([i for j in self.set_of_cities for i in j.results]))

        for i in best_algs:
            # Results of algorithm i on all the problems
            df = pd.DataFrame(dict([(index, j[i]) for index, j in enumerate(normalised_results) if i in j.columns]))
            df.ffill(axis=0, inplace=True)
            try:
                ax.step(df.index, np.mean(df, axis=1), where='post', label=i)
            except ValueError:
                print(normalised_results[0].index)
                print(normalised_results[1].index)
                print(normalised_results[2].index)
                print(list(df.index))
                print(df.loc[:20])
                raise
            np.seterr(all='ignore')
            conf_interval = st.norm.interval(confidence=0.95, loc=np.mean(df, axis=1), scale=st.sem(df,axis=1))
            np.seterr(all='raise')
            # ax.fill_between(df.index, np.min(df, axis=1), np.max(df, axis=1), alpha=0.2)
            ax.fill_between(df.index, conf_interval[0], conf_interval[1], alpha=0.2)

        ax.legend()

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def plot_rank_topN(self, axes_or_filename, best_algs, xlogscale=False, ylogscale=False):

        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        if xlogscale:
            ax.set_xscale('log')

        if ylogscale:
            ax.set_yscale('log')

        results_ranks = []

        # Executed algs
        algs = list(set([i for j in self.set_of_cities for i in j.results]))

        ax.set_ylim(1,len(algs))

        for i_problem in self.set_of_cities:
            data_results = pd.DataFrame(
                dict([(key, pd.Series(dict(value))) for key, value in i_problem.results.items()]))

            for i_alg in algs:
                if i_alg not in data_results.columns:
                    data_results = pd.concat((data_results, pd.DataFrame({i_alg: dict([(1,i_problem.size()*TSP_Instance.sqrt_2)])})), axis=1)

            data_results.ffill(axis=0, inplace=True)
            data_results = data_results.round(decimals=6)
            results_ranks.append(data_results.rank(axis=1))

        for i in best_algs:
            df = pd.DataFrame({str(index): j[i] for index, j in enumerate(results_ranks)})
            df.ffill(axis=0, inplace=True)
            mean = np.mean(df, axis=1)
            std = np.std(df, axis=1)
            np.seterr(all='ignore')
            conf_interval = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(df,axis=1))
            np.seterr(all='raise')
            ax.step(df.index, mean, where='post', label=i)
            # ax.fill_between(df.index, mean - std, mean + std, alpha=0.2)
            ax.fill_between(df.index, conf_interval[0], conf_interval[1], alpha=0.2)

        ax.legend()

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()