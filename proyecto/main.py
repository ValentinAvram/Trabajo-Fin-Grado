from Test_frame.test_frame import test_frame
from Algorithms.common_function_algorithms import read_json

"""
    Base de la ejecición de los algoritmos, importación de librerías y definición de variables globales
"""
def main():
    param = read_json()
    number_problems = param["number_problems"]
    runtime = param["max_runtime"]
    min_cities = param["min_cities"]
    max_cities = param["max_cities"]

    test_frame(number_problems, runtime, min_cities, max_cities)

    return None


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(e)
    exit(0)
