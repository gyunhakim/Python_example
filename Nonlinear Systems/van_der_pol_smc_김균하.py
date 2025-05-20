"""
Example 3.7
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import time


def van_der_pol_smc(time, state, mu, c1, c2, k):
    s = c1 * state[0] + c2 * state[1]
    u = (-c1 * state[1] / c2 - mu * (1 - state[0]**2) * state[1] + state[0]
         - k * np.sign(s))

    return [
        state[1],
        -state[0] + mu * (1.0 - state[0]**2) * state[1] + u
    ]


# Main function
def run_sim():
    args = mu, c1, c2, k = 8, 3, 1, 1
    t_start, t_end, t_step = 0.0, 10.0, 0.001

    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end + t_step, t_step)

    state_init = [-1.5, 3]

    # solver_choice = "odeint"
    solver_choice = "solve_ivp"

    try:
        start = time.perf_counter()
        if solver_choice == "odeint":
            states, infodict = odeint(
                van_der_pol_smc, state_init, t_eval, args,
                tfirst=True, full_output=True,
            )
            states = states.T
            times = t_eval
            # Check to see if there are numerical problems
            if infodict["message"] != "Integration successful.":
                print("\nNumerical problems arise.\n")
                return
        else:
            sol = solve_ivp(
                van_der_pol_smc, t_span, state_init, args=args,
                t_eval=t_eval
            )
            states = sol.y
            times = sol.t
            # Check to see if there are numerical problems
            if not sol.success:
                print("\nNumerical problems arise.\n")
                return

        comp_time = 1.0 * (time.perf_counter() - start)

    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
        return

    s = c1 * states[0] + c2 * states[1]
    u = (-c1 * states[1] / c2 - mu * (1 - states[0]**2) * states[1] + states[0]
         - k * np.sign(s))

    print(f"\nComputation time: {comp_time:>.2f}sec.\n")

    # 결과 그래프 (Phase Portrait)
    plt.figure(figsize=(12, 6))
    plt.plot(states[0], states[1], label='Phase Portrait')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Van der Pol Oscillator Phase Portrait')
    # plt.show()

    # 결과 그래프 (Phase Portrait)
    plt.figure(figsize=(12, 6))
    plt.plot(states[0], states[1], label='Phase Portrait')
    plt.plot(states[0], (-c1 / c2) * states[0], 
            linestyle='--', label='Sliding Surface')
    plt.quiver(states[0], states[1], 0.1 * u, 0, 
            scale=30, color='red', label='Control Input')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Van der Pol Oscillator Phase Portrait with Sliding Surface')
    # plt.show()

    # 결과 그래프 (Control Input)
    plt.figure(figsize=(12, 6))
    plt.plot(times, u, label='Control Input', color='red')
    plt.xlabel('Time')
    plt.ylabel('Control Input')
    plt.legend()
    plt.title('Control Input over Time')
    # plt.show()

    # 결과 그래프 (Time Response)
    plt.figure(figsize=(12, 6))
    plt.plot(times, states[0], label='x1', color='blue')
    plt.plot(times, states[1], label='x2', color='green')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.title('Time Response')

    plt.show(block=True)
    # _ = input("Press Enter to Finish. ")
    # print()

if __name__ == "__main__":
    run_sim()
