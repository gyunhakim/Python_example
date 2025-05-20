"""
Solving a 2nd order DE and plotting the phase portrait (Ex 2.11)
(by T.-W. Yoon, Oct. 2023)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def read_two_numbers(keyboard_message):
    """
    Print a message and read two real numbers from keyboard
    """
    nos = input(keyboard_message).split()
    while True:
        if len(nos) == 2:
            try:
                two_numbers = (float(nos[0]), float(nos[1]))
            except ValueError:
                pass
            else:
                break
        nos = input("two numbers please: ").split()

    return two_numbers


def streamplot(func, ax, axis, lw=1.0, density=2.0):
    """
    Creates a streamplot representing the direction and intensity
    of a vector field.

    Parameters:
    - func: Function that computes the derivatives at each point
    - ax: Axes object for plotting
    - axis: List or tuple specifying the x and y axis limits
    - lw: Line width for the streamplot (default = 1.0)
    - density: Spacing between streamlines (default = 2.0).

    Returns: None
    """
 
    # Adjust the number of grid cells with the density of the streamplot
    no_of_data = round(density * 30)
    x1 = np.linspace(axis[0], axis[1], no_of_data)
    x2 = np.linspace(axis[2], axis[3], no_of_data)
    x1_data, x2_data = np.meshgrid(x1, x2)
    x1_prime, x2_prime = func(None, [x1_data, x2_data])

    # speed = np.sqrt(x1_prime**2 + x2_prime**2)
    # lw = 5 * speed / speed.max()

    ax.streamplot(
        x1_data, x2_data, x1_prime, x2_prime,
        density=density, color="y",
        linewidth=lw
    )
    ax.set_xlim([axis[0], axis[1]])
    ax.set_ylim([axis[2], axis[3]])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")


# Differential equation in question
def nonlinear_sys(t, x):
    return [
        -x[0] + x[0] * x[1],
        x[0] + x[1] - 2 * x[0] * x[1]
    ]


def main():
    # Set the axis for the phase portraits
    axis = [-2, 3, -2, 3]

    # Set the figure for state streams
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=100)
    streamplot(nonlinear_sys, ax, axis, lw=0.8)
    ax.plot(0, 0, "o")  # equilibrium
    ax.plot(1, 1, "o")  # equilibrium
    ax.set_aspect('equal', 'box')
    ax.set_title("State streams")

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    plt.show(block=False)
    plt.pause(0.5)

    while True:
        # Read two numbers for the initial states
        x0 = read_two_numbers("Initial condition: (Type '0 0' to stop) ")

        # Entering '0 0' will exit the code
        if x0 == (0, 0):
            break

        # Set the parameters for dynamic simulations
        t_init, t_final = 0, 10
        t_span = t_init, t_final
        No_of_samples = 100
        t_eval = np.linspace(t_init, t_final, No_of_samples + 1)

        # Sove the ODE
        try:
            sol = solve_ivp(
                nonlinear_sys,
                t_span,
                x0,
                t_eval=t_eval,
            )
            if not sol.success:
                print("\nNumerical problems arise.\n")
                continue
        except Exception as e:  # Exception handling
            print(f"\nAn error occurred: {e}\n")
            continue

        plt.close(fig)  # Close the previous figure
        # Set the figure for the phase portrait and time solutions
        fig, ax = plt.subplots(1, 2, figsize=(15, 7), dpi=100)

        # Plot the phase portrait with state streams
        streamplot(nonlinear_sys, ax[0], axis, lw=0.8)
        ax[0].plot(sol.y[0], sol.y[1], "b-")  # path
        ax[0].plot(0, 0, "o")
        ax[0].plot(1, 1, "o")  # equilibrium
        ax[0].plot([sol.y[0, 0]], [sol.y[1, 0]], "s")
        ax[0].set_aspect('equal', 'box')
        ax[0].set_title("Phase portrait")

        # Plot the time solution
        ax[1].plot(sol.t, sol.y[0], "b-")
        ax[1].plot(sol.t, sol.y[1], "g-")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("State variables")
        ax[1].set_title("Time responses")

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        plt.show(block=False)
        plt.pause(0.5)


if __name__ == "__main__":
    main()
