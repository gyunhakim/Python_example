"""
Simulation of an RLC circuit using solve_ivp from scipy
Lyapunov function is obtained and plotted
"""


# RLC 직렬회로 방정식
def rlc_eqn(time, state, *args):
    resistor, inductor, capacitor, voltage = args

    # state[0]: 카페시터 전압, state[1]: 전류
    return [
        state[1] / capacitor,
        -state[0] / inductor - (resistor / inductor) * state[1] + voltage / inductor,
    ]


# 메인 함수
def run_sim():
    import numpy as np  # 벡터/행렬 등 수치 계산용 패키지
    import matplotlib.pyplot as plt  # 그림 그리기 패키지
    from scipy.integrate import solve_ivp  # 미분방정식 푸는 함수
    from scipy.linalg import solve_lyapunov  # 리아프노프 방정식 푸는 함수
    from scipy.linalg import cholesky

    t_start = 0.0  # 시뮬레이션 시작하는 시간
    t_end = 10.0  # 시뮬레이션 끝나는 시각
    t_step = 0.1  # 결과 출력 시간 간격
    no_steps = round((t_end - t_start) / t_step) + 1

    # 시간 설정 (t_start, t_start+t_step, ..., t_end)
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, no_steps)

    state_init = [1.0, 1.0]  # 초깃값 설정
    args = resistor, inductor, capacitor, voltage = 1, 2, 1, 0

    try:
        # 미분방정식 풀기
        sol = solve_ivp(rlc_eqn, t_span, state_init, t_eval=t_eval, args=args)

        # 수치 오류가 있는지 확인
        if not sol.success:
            print("\nNumerical problems arise.\n")
            return

    except Exception as e:
        # 예외가 발생하면 그 내용을 출력
        print(f"\nAn error occurred: {e}\n")
        return

    # 리아프노프 방정식 A^T P + P A = -I 풀기
    a_matrix = np.array([[0, 1 / capacitor], [-1 / inductor, -resistor / inductor]])
    p_matrix = solve_lyapunov(a_matrix.T, -np.eye(2))
    eigenvalues, eigenvectors = np.linalg.eig(p_matrix)

    largest_index = np.argmax(eigenvalues.real)
    smallest_index = 1 - largest_index

    p_sqrt = cholesky(p_matrix, lower=False)  # p_matrix = p_sqrt.T @ p_sqrt
    p_sqrt_inv = np.linalg.inv(p_sqrt)

    # 리아프노프 함수 x^T P x와 x^T x 계산
    xpx = (
        p_matrix[0, 0] * sol.y[0] ** 2
        + 2 * p_matrix[0, 1] * sol.y[0] * sol.y[1]
        + p_matrix[1, 1] * sol.y[1] ** 2
    )
    xx = np.linalg.norm(sol.y, axis=0) ** 2  # sol.y[0]**2 + sol.y[1]**2

    r = 1.0
    theta = np.linspace(0, 2 * np.pi, 100)
    px = np.array([r * np.cos(theta), r * np.sin(theta)])
    x_vectors = p_sqrt_inv @ px

    plt.rcParams.update({"font.size": 8})
    fig = plt.figure(figsize=(9, 7), dpi=100)
    ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)
    ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, fig=fig)

    ax1.plot(sol.t, sol.y[0], label="$x_1$")
    ax1.plot(sol.t, sol.y[1], label="$x_2$")
    ax1.legend(loc="best")
    ax1.set_title("Time responses")
    # ax1.set_title("RLC circuit")
    ax2.plot(sol.t, xx, label="$x^T x$")
    ax2.plot(sol.t, xpx, label="$x^T P x$")
    ax2.legend(loc="best")
    ax2.set_xlabel("Time")
    ax3.plot(x_vectors[0], x_vectors[1])

    for index in (smallest_index, largest_index):
        x, y = eigenvectors[0, index], eigenvectors[1, index]
        if index == smallest_index:
            r = 1 / np.sqrt(eigenvalues[index])
            color = 'red'
        else:
            r = 1 / np.sqrt(eigenvalues[index])
            color = 'blue'
        ax3.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc=color, ec=color)
        x, y = r * np.cos(theta), r * np.sin(theta)
        ax3.plot(x, y, linestyle='dotted', color=color)
        
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_aspect('equal', 'box')
    ax3.set_xlabel("\nred (blue): eigenvector associated with the smallest (largest) eigenvalue")
    ax3.set_title("$x^T P x = 1$")
    plt.show(block=True)


if __name__ == "__main__":
    run_sim()
