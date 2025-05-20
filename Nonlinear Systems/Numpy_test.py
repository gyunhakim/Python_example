# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:38:47 2020

"""


def main():
    import numpy as np
    import time

    def sample_function(x):
        return np.sin(x) + np.cos(x)

    # sample_function = lambda x: np.sin(x) + np.cos(x)

    no_data = 1001
    x_values = np.linspace(-np.pi, np.pi, no_data)

    time_conv = 10**3

    start = time.perf_counter()
    func_eval = sample_function(x_values)
    ans = time_conv * (time.perf_counter() - start)
    print(f"computation time for vectorization: {ans:>.2e} (msec)")

    values_iterated = np.zeros(len(x_values))

    start = time.perf_counter()
    for index, value in enumerate(x_values):
        values_iterated[index] = sample_function(value)
    ans = time_conv * (time.perf_counter() - start)
    print(f"computation time for loop: {ans:>.2e} (msec)")

    results_identical = np.allclose(func_eval, values_iterated)
    print(f"\nDo these lead to the same results? {results_identical}")


if __name__ == "__main__":
    main()
