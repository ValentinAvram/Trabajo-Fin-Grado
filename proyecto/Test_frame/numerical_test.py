from scipy import stats
#import Orange3
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
from scipy.stats import rankdata, chi2


def wilcoxon_test(values_alg1, values_alg2, type):
    t_stat, p_value = stats.wilcoxon(values_alg1, values_alg2)

    alfa_t_test = 0.05

    if p_value < alfa_t_test:
        var = f'Wilcoxon {type}: Diferencias significativas. T-value:{t_stat}, p-value_ {p_value} con alfa: 0.05 \n'
    else:
        var = f'Wilcoxon {type}: Sin diferencias significativas. T-value:{t_stat}, p-value_ {p_value} con alfa: 0.05 \n'

    return var


def friedmann_test(values, algs_names, caso):
    estadistico, p_value = stats.friedmanchisquare(*values)

    alpha = 0.05  # Nivel de significancia

    cadena = ' vs '.join(algs_names)

    if p_value < alpha:
        var = f'Friedman caso: {caso}. Algoritmos: {cadena}. Hay diferencias significativas entre los algoritmos. Estadístico: {estadistico}, p-value: {p_value} con alfa: {alpha} \n'
    else:
        var = f'Friedmann caso: {caso}. Algoritmos: {cadena}. No hay diferencias significativas entre los algoritmos. Estadístico: {estadistico}, p-value: {p_value} con alfa: {alpha} \n'

    return var

def nemenyi_test(values, algs_names, caso):
    n_datasets = len(values[0])

    data = Orange.data.Table(np.array(values))
    cd = Orange.evaluation.compute_CD(data, n_datasets, alpha='0.05', test="nemenyi")

    rangos = []
    values = np.transpose(values)
    for problema in values:
        rangos.append(rankdata(problema).tolist())

    ranks = list(map(mean, zip(*rangos)))
    Orange.evaluation.graph_ranks(ranks, algs_names, cd=cd, width=10, textspace=1.5)

    plt.annotate(f'CD = {cd:.2f}', xy=(0.5, 0.02), xycoords='axes fraction', fontsize=10, color='black')

    name = f'data/nemenyi_{caso}.png'
    plt.savefig(name)
    plt.close()

    return str(cd)

def friedman_test_manual(lista_algs):
    # K es numero de algoritmos y N es numero de problemas
    k = len(lista_algs)
    n = len(lista_algs[0])

    rangos_promedio = [np.mean(r) for r in lista_algs]

    R = sum([r ** 2 for r in rangos_promedio])

    friedman_statistic = (12 * n * (R - (k * (k + 1) ** 2) / 4)) / (k * (k + 1))

    print(f"Estadístico de Friedman: {friedman_statistic}")

    grados_libertad = k - 1
    p_value = 1 - chi2.cdf(friedman_statistic, df=grados_libertad)

    print(f"Valor p: {p_value}")
    if p_value < 0.05:
        texto = "Diferencias significativas"
    else:
        texto = "Sin diferencias significativas"

    return friedman_statistic, p_value, texto