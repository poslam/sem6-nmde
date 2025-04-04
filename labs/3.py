import numpy as np
from scipy.sparse import diags


def exact(x, t):
    return np.exp(-np.pi * t) * np.sin(np.pi * x / 4)


def heat_equation_explicit(L, T, Nx, Nt, alpha, u_init, mu, beta):
    dx = L / Nx
    dt = T / Nt
    r = alpha * dt / dx**2

    if r > 0.5:
        print("Предупреждение: схема может быть неустойчивой")

    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = u_init(x)

    for n in range(0, Nt):
        for i in range(1, Nx):
            u[n + 1, i] = u[n, i] + r * (u[n, i - 1] - 2 * u[n, i] + u[n, i + 1])

        u[n + 1, 0], u[n + 1, -1] = mu(t[n + 1]), beta(t[n + 1])

    return u


def heat_equation_cn(L, T, Nx, Nt, alpha, u_init, mu, beta):
    dx = L / Nx
    dt = T / Nt
    r = alpha * dt / (2 * dx**2)

    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = u_init(x)

    A = diags([-r, 1 + 2 * r, -r], [-1, 0, 1], shape=(Nx - 1, Nx - 1)).toarray()
    B = diags([r, 1 - 2 * r, r], [-1, 0, 1], shape=(Nx - 1, Nx - 1)).toarray()

    for n in range(0, Nt):
        b = B @ u[n, 1:Nx]
        u[n + 1, 1:Nx] = np.linalg.solve(A, b)
        u[n + 1, 0], u[n + 1, -1] = mu(t[n + 1]), beta(t[n + 1])

    return u


L = 4
T = 1
N = 10
M = 100
alpha = 16 / np.pi

x = np.linspace(0, L, N + 1)
t = np.linspace(0, T, M + 1)

u_init = lambda x: np.sin(np.pi * x / 4)
mu = lambda t: 0
beta = lambda t: 0


u_explicit = heat_equation_explicit(L, T, N, M, alpha, u_init, mu, beta)
u_cn = heat_equation_cn(L, T, N, M, alpha, u_init, mu, beta)

X, T_grid = np.meshgrid(x, t)
u_exact = exact(X, T_grid)

error_explicit = np.abs(u_explicit - u_exact)
error_cn = np.abs(u_cn - u_exact)

print(np.max(error_explicit), np.max(error_cn))
