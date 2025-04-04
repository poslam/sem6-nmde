import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def solve_hyperbolic_eq(a, l, T, h, tau, phi, psi, mu1, mu2, f=lambda x, t: 0):
    """
    Решает волновое уравнение с помощью явной разностной схемы.

    Параметры:
        a: скорость волны (a > 0)
        l: длина струны (пространственный интервал [0, l])
        T: конечное время
        h: шаг по пространству
        tau: шаг по времени
        phi: начальное условие u(x, 0) = phi(x)
        psi: начальная скорость du/dt(x, 0) = psi(x)
        mu1, mu2: граничные условия u(0, t) = mu1(t), u(l, t) = mu2(t)
        f: функция источника f(x, t) (по умолчанию 0)
    """

    Nx = int(l / h) + 1
    Nt = int(T / tau) + 1
    x = np.linspace(0, l, Nx)
    t = np.linspace(0, T, Nt)

    u = np.zeros((Nt, Nx))

    for i in range(Nx):
        u[0, i] = phi(x[i])

    for i in range(1, Nx - 1):
        u[1, i] = (
            u[0, i]
            + tau * psi(x[i])
            + (a**2 * tau**2 / (2 * h**2)) * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])
            + (tau**2 / 2) * f(x[i], 0)
        )

    u[1, 0] = mu1(t[1])
    u[1, -1] = mu2(t[1])

    for n in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            u[n + 1, i] = (
                2 * u[n, i]
                - u[n - 1, i]
                + (a**2 * tau**2 / h**2) * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
                + tau**2 * f(x[i], t[n])
            )

        u[n + 1, 0] = mu1(t[n + 1])
        u[n + 1, -1] = mu2(t[n + 1])

    return x, t, u


a = 1
l = 1
T = 1
N = 10
M = 10

h = l / N
tau = T / M

# вариант 5

phi = lambda x: 0
psi = lambda x: 0
mu1 = lambda t: t**2 - t
mu2 = lambda t: 3 * t**2
f = lambda x, t: 0

# вариант 10

# phi = lambda x: 0
# psi = lambda x: 0
# mu1 = lambda t: -3 * t**2
# mu2 = lambda t: -2 * t
# f = lambda x, t: 0

# вариант 11

# phi = lambda x: 0
# psi = lambda x: 0
# mu1 = lambda t: -t
# mu2 = lambda t: t**3
# f = lambda x, t: 0

# вариант 12

# phi = lambda x: 0
# psi = lambda x: 0
# mu1 = lambda t: 2 * t * (t - 1)
# mu2 = lambda t: 2 * t**2
# f = lambda x, t: 0

# вариант 13

# phi = lambda x: x * (1 - x)
# psi = lambda x: x**3 - x**2
# mu1 = lambda t: 0
# mu2 = lambda t: 0
# f = lambda x, t: t * x**2 * (1 - x)

x, t, u = solve_hyperbolic_eq(a, l, T, h, tau, phi, psi, mu1, mu2, f)

fig, ax = plt.subplots()
(line,) = ax.plot(x, u[0, :], "b-", lw=2)
ax.set_xlim(0, l)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Волновое уравнение: численное решение")

print(len(t))

start_frame = 0


def update(frame):
    actual_frame = (start_frame + frame) % len(t)
    line.set_ydata(u[actual_frame, :])
    ax.set_title(f"Волновое уравнение: t = {t[actual_frame]:.2f}")
    return (line,)


line.set_ydata(u[start_frame, :])

ani = FuncAnimation(fig, update, frames=len(t), interval=50)


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# X, T = np.meshgrid(x, t)

# surf = ax.plot_surface(X, T, u, cmap="viridis")

# ax.set_xlabel("x")
# ax.set_ylabel("t")
# ax.set_zlabel("u(x,t)")
# ax.set_title("Волновое уравнение: трехмерное решение")

# ax.view_init(elev=30, azim=45)

plt.show()
