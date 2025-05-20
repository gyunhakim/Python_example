"""
Simulation of an RLC circuit using odeint from scipy
"""


# RLC 직렬회로 방정식
def rlc_eqn(time, state, *args):
    resistor, inductor, capacitor, voltage = args

    # state[0]: 카페시터 전압, state[1]: 전류
    return [
      state[1] / capacitor,
      -state[0] / inductor - (resistor / inductor) * state[1] + voltage / inductor
    ]


# 메인 함수
def run_sim():
    import numpy as np  # 벡터/행렬 등 수치 계산용 패키지
    import matplotlib.pyplot as plt  # 그림 그리기 패키지
    from scipy.integrate import odeint  # 미분방정식 푸는 함수

    t_start = 0.0  # 시뮬레이션 시작하는 시간
    t_end = 10.0   # 시뮬레이션 끝나는 시각
    t_step = 0.1   # 결과 출력 시간 간격
    no_steps = round((t_end - t_start) / t_step) + 1

    # 시간 설정 (t_start, t_start+t_step, ..., t_end)
    t_eval = np.linspace(t_start, t_end, no_steps)
    state_init = [0.0, 0.0]  # 초깃값 설정
    args = R, L, C, V = 1, 1, 1, 1  # 입력 전압 V와  R, L, C 값은 모두 1이라 가정

    try:
        # 미분방정식 풀기
        states, infodict = odeint(
            rlc_eqn, state_init, t_eval, args,
            tfirst=True, full_output=True,
        )
        # 수치 오류가 있는지 확인
        if infodict["message"] != "Integration successful.":
            print("\nNumerical problems arise.\n")
            return

    except Exception as e:
        # 예외가 발생하면 그 내용을 출력
        print(f"\nAn error occurred: {e}\n")
        return

    # print(f"\nData shapes of t_eval and states: {t_eval.shape}, {states.shape}\n")
    plt.plot(t_eval, states)
    plt.show()  # 결과 그리기


if __name__ == "__main__":
    run_sim()
