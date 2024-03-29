from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as st

# define Python user-defined exceptions
class MaxRuntimeHit(Exception):
    "Raised when the input value is less than 18"
    pass

class TSP_Instance:

    sqrt_2 = np.sqrt(2)

    def __init__(self, N, max_runtime):
        self.N = N
        self.cities = TSP_Instance.random_cities(N)
        self.results = {}
        self.distances = np.zeros((N, N))
        self.max_runtime = max_runtime

        self._num_evals = 0
        self._init_time = None

        x = self.cities['x']
        y = self.cities['y']

        for i in range(N):
            self.distances[i, i] = 0
            for j in range(i + 1, N):
                self.distances[i, j] = np.trunc(10000 * np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)) / 10000
                self.distances[j, i] = self.distances[i, j]

    def __str__(self):
        output = '-----TSP INSTANCE BEGIN--\n'
        output += str(self.N) + '\n'
        output += str(self.cities) + '\n'
        output += str(self.distances) + '\n'
        output += str(self.results) + '\n'
        output += str(self.max_runtime) + '\n'
        output += '-----TSP INSTANCE END__--\n'
        return output

    def clear_results(self):
        self.results = {}

    def get_results(self, alg):
        if alg in self.resutls:
            return self.results[alg]
        else:
            return [(1,self.sqrt_2 * self.size())]

    @classmethod
    def random_cities(cls, N):
        # generate N random 2D coordinates
        x = np.random.rand(N)
        y = np.random.rand(N)

        return {'x': x, 'y': y}

    def size(self):
        return len(self.cities['x'])

    def plot_cities(self, axes_or_filename):
        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        ax.scatter(self.cities['x'], self.cities['y'])

        for i, (x, y) in enumerate(zip(self.cities['x'], self.cities['y'])):
            ax.text(x + 0.01, y, str(i))

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def get_num_evals(self):
        return self._num_evals

    def _reset_num_evals(self, max_evals=None):
        self._num_evals = 0

        if max_evals is not None:
            self.max_evals = max_evals

    def _reset_init_time(self):
        self._init_time = datetime.now()

    def cost(self, permutation):

        # print(permutation)
        if 0 in permutation:
            permutation_aux = permutation
        else:
            permutation_aux = [i-1 for i in permutation]

        x = self.cities['x']
        y = self.cities['y']

        # calculate the total distance traveled in the circuit
        total_distance = 0
        for i in range(len(permutation_aux) - 1):
            p1 = permutation_aux[i]
            p2 = permutation_aux[i + 1]
            distance = self.distances[p1, p2]  # np.sqrt((x[p1]-x[p2])**2 + (y[p1]-y[p2])**2)
            # print(distance)
            total_distance += distance

        # add the distance from the last point to the first point
        p1 = permutation_aux[-1]
        p2 = permutation_aux[0]
        distance = self.distances[p1, p2]  # np.sqrt((x[p1]-x[p2])**2 + (y[p1]-y[p2])**2)
        total_distance += distance

        # return the total distance traveled
        return total_distance

    def draw_circuit(self, permutation, axes_or_filename, line_format='r--', linewidth=1, alpha=0.5):
        # set the number of points
        N = len(permutation)

        # create a scatter plot of the points
        x = self.cities['x']
        y = self.cities['y']
        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, axes = plt.subplots()
        else:
            axes = axes_or_filename
        axes.scatter(x, y)

        # create a list of x and y coordinates in the order specified by the permutation
        x_ordered = [x[i] for i in permutation]
        y_ordered = [y[i] for i in permutation]

        # add a line connecting each point in the order specified by the permutation
        for i in range(N - 1):
            axes.plot([x_ordered[i], x_ordered[i + 1]],
                      [y_ordered[i], y_ordered[i + 1]],
                      line_format, linewidth=linewidth, alpha=alpha)

        # add a line connecting the last point to the first point
        axes.plot([x_ordered[N - 1], x_ordered[0]],
                  [y_ordered[N - 1], y_ordered[0]],
                  line_format, linewidth=linewidth, alpha=alpha)

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()

    def evaluate_and_resgister(self, permutation, alg_id):

        current_cost = self.cost(permutation)

        if not alg_id in self.results:
            self._reset_init_time()
            self._reset_num_evals()
            self.results[alg_id] = [(1, current_cost)]
        else:
            # now_time = datetime.now()
            #
            # if now_time - self._init_time > timedelta(seconds=self.max_runtime):
            #     raise MaxRuntimeHit

            self._num_evals += 1
            value_to_be_registered = min(self.results[alg_id][-1][1], current_cost)

            if value_to_be_registered < self.results[alg_id][-1][1]:
                self.results[alg_id].append((self._num_evals, value_to_be_registered))

        return current_cost

    def plot_convergence_graphs(self, axes_or_filename):
        if not type(axes_or_filename) == Axes:
            if not type(axes_or_filename) == str:
                print('Wrong type of axes in draw_circuit. Exitting function')
                return
            else:
                fig, ax = plt.subplots()
        else:
            ax = axes_or_filename

        for i in self.results:
            x = [j[0] for j in self.results[i]]
            y = [j[1] for j in self.results[i]]
            ax.step(x,y,where='post')

        if type(axes_or_filename) == str:
            plt.savefig(axes_or_filename)
            plt.close()


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