"""
Example 3.7
"""

import numpy as np
import os
import json


def system_eqn(t, x, a, b, c):
    # x[0], x[1]: two state variables
    # x[2] = dx[0]/da, x[3] = dx[1]/da
    # x[4] = dx[0]/db, x[5] = dx[1]/db
    # x[6] = dx[0]/dc, x[7] = dx[1]/dc

    return [
        x[1],
        -c * np.sin(x[0]) - b * x[1] * np.cos(x[0]) - a * x[1],
        x[3],
        -c * x[2] * np.cos(x[0]) + b * x[1] * x[2] * np.sin(x[0])
        - a * x[3] - b * x[3] * np.cos(x[0]) - x[1],
        x[5],
        -c * x[4] * np.cos(x[0]) + b * x[1] * x[4] * np.sin(x[0])
        - a * x[5] - b * x[5] * np.cos(x[0]) - x[1] * np.cos(x[0]),
        x[7],
        -c * x[6] * np.cos(x[0]) + b * x[1] * x[6] * np.sin(x[0])
        - a * x[7] - b * x[7] * np.cos(x[0]) - np.sin(x[0]),
    ]


# Input the parameters from a json file
def load_parameters_from_file(file_path):
    with open(file_path, "r") as config_file:
        parameters = json.load(config_file)
    return parameters


# Main function
def run_sim():
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp  # ODE Solver

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "Ex_3.7_parameters.json")

    try:
        parameters = load_parameters_from_file(json_file)
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
        return

    # Parameters in question
    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]

    # a, b, c = 1, 0, 1

    # Initial conditions
    x1 = parameters["x1"]  # Initial position of the cart
    x2 = parameters["x2"]  # Initial velocity of the cart

    # x1, x2 = 1.0, 1.0

    # Time span
    t_start = parameters["t_start"]
    t_end = parameters["t_end"]
    t_step = parameters["t_step"]

    # t_start, t_end, t_step = 0.0, 10.0, 0.1

    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end + t_step, t_step)

    state_init = np.zeros(8)
    state_init[:2] = [x1, x2]

    try:
        # Solve the ODE
        sol = solve_ivp(system_eqn, t_span, state_init,
                        t_eval=t_eval, args=(a, b, c))

        # Check if there are numerical problems
        if not sol.success:
            print("\nNumerical problems arise.\n")
            return

    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
        return

    plt.rcParams.update({"font.size": 8})
    _, ax = plt.subplots(2, 2, figsize=(9, 7), dpi=100, sharex=True)

    # ax[0, 0].plot(sol.t, sol.y[0], label="$x_1$")
    # ax[0, 1].plot(sol.t, sol.y[1], label="$x_2$")
    # ax[1, 0].plot(sol.t, sol.y[2], label="$\partial x_1/\partial a$")
    # ax[1, 0].plot(sol.t, sol.y[4], label="$\partial x_1/\partial b$")
    # ax[1, 0].plot(sol.t, sol.y[6], label="$\partial x_1/\partial c$")
    # ax[1, 1].plot(sol.t, sol.y[3], label="$\partial x_2/\partial a$")
    # ax[1, 1].plot(sol.t, sol.y[5], label="$\partial x_2/\partial b$")
    # ax[1, 1].plot(sol.t, sol.y[7], label="$\partial x_2/\partial c$")
    # ax[0, 0].legend(loc="best")
    # ax[0, 1].legend(loc="best")
    # ax[1, 0].legend(loc="best")
    # ax[1, 1].legend(loc="best")

    data = [
        (0, 0, [0], ["$x_1$"]),
        (0, 1, [1], ["$x_2$"]),
        (
            1,
            0,
            [2, 4, 6],
            [
                "$\partial x_1/\partial a$",
                "$\partial x_1/\partial b$",
                "$\partial x_1/\partial c$",
            ],
        ),
        (
            1,
            1,
            [3, 5, 7],
            [
                "$\partial x_2/\partial a$",
                "$\partial x_2/\partial b$",
                "$\partial x_2/\partial c$",
            ],
        ),
    ]

    for i, j, indices, labels in data:
        for index, label in zip(indices, labels):
            ax[i, j].plot(sol.t, sol.y[index], label=label)
        ax[i, j].legend(loc="best")

    plt.show(block=True)  # Plot the results


if __name__ == "__main__":
    run_sim()
