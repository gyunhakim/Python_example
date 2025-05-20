# import numpy as np
# import matplotlib.pyplot as plt

# def sign(x):
#     return (x > 0) - (x < 0) or 0

sign = lambda x: (x > 0) - (x < 0) or 0

print(sign(0))


# times = np.linspace(0, 2 * np.pi, 100)
# sin_t = np.sin(times)
# cos_t = np.cos(times)

# plt.subplot(2, 1, 1)
# plt.plot(times, sin_t)
# plt.ylabel("$\sin t$")
# plt.subplot(2, 1, 2)
# plt.plot(times, cos_t)
# plt.ylabel("$\cos t$")
# plt.show()

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(times, sin_t)
# ax[0].set_ylabel("$\sin t$")
# ax[1].plot(times, cos_t)
# ax[1].set_ylabel("$\cos t$")
# plt.show()  # st.pyplot(fig)

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
# ax1.plot(times, sin_t)
# ax1.set_ylabel("$\sin t$")
# ax2.plot(times, cos_t)
# ax2.set_ylabel("$\cos t$")
# plt.show()  #st.pyplot(fig)

# switch = 2

# result = 'first' if switch == 1 else 'second' if switch == 2 else 'etc'

# if switch == 1:
#     result = 'first'
# elif switch == 2:
#     result = 'second'
# else:
#     result = 'etc'

# result = 'first' if switch == 1 else 'not first'

# if switch == 1:
#     result = 'first'
# else:
#     result = 'not first'
# print(result)

# even_list = []
# for index in range(10):
#     if index % 2 == 0:
#         even_list.append(index)

# even_list = [index for index in range(10) if index % 2 == 0]
# print(even_list)

# t_start, t_end, t_step = 0.0, 1.0, 0.1
# no_steps = round((t_end - t_start) / t_step) + 1
# t_span = np.linspace(t_start, t_end, no_steps)
# t_span = np.arange(t_start, t_end + t_step, t_step)
# print(t_span)
