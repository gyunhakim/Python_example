import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Define the system dynamics

def dynamics(t, state, *args):
    a, r, epsilon = args
    x1, x2 = state
    dx1dt = -(x2-a)*x1 + epsilon
    dx2dt = r*x1**2 
    # dx2dt = r*np.abs(x1) 
    # dx2dt = r*np.abs(x1)**3

    return [dx1dt, dx2dt]

def run_sim():
    t_start = 0.0
    t_end = 10.0
    t_span = [t_start, t_end]

    x1_init = 4*np.random.rand() - 2  # -2~2 사이의 초기값 x1_init
    x2_init = 2*np.random.rand()  # 0~2 사이의 초기값 x2_init

    state_init = [x1_init, x2_init]
    a = 4 # 0~5 사이의 숫자
    # a = 5*np.random.rand() # 0~5 사이의 숫자
    # b = 5
    r = 2
    epsilon = 1.e-2
    args = a, r, epsilon

    try:
        sol = solve_ivp(dynamics, t_span, state_init, dense_output=True, args=args)

    except Exception as e:
        # 예외가 발생하면 그 내용을 출력
        print(f"\nAn error occurred: {e}\n")
        return

    # f_lyapunov = 0.5*(np.power(sol.y[0,:],2) + np.power((sol.y[1,:] - (a + epsilon)),2)/r)
    plt.plot(sol.t, sol.y[0,:],label='x1')
    plt.plot(sol.t, sol.y[1,:],label='x2')
    # plt.plot(sol.t, f_lyapunov,label='Lyapynov Function')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('State Variable & Lyapunov F')
    plt.show()  # 결과 그리기


if __name__ == "__main__":
    run_sim()
    a=1