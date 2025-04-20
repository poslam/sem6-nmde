from typing import Union

import numpy as np
from tabulate import tabulate


def print_matrix(matrix: np.ndarray, header: str = None):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    s = f"{tabulate(str_matrix, tablefmt='fancy_grid' ,)}\n"

    if header is not None:
        header = str(header).center(len(s.split("\n")[0]))
        print(header)

    print(s)


def check_ans(M: np.ndarray, ans: np.ndarray) -> bool:
    print_matrix(M[:, :-1] @ ans - M[:, -1])


def check_eigvec(A: np.ndarray, vec: np.ndarray, val: float):
    print_matrix((A @ vec) / val - vec, header="check eigvec")


def check_eigvals(A: np.ndarray, vals: Union[list[np.ndarray]]):
    print("check eigvals with det(A - val * E):")
    n = A.shape[0]

    for i in range(len(vals)):
        print(f"{i}: {np.linalg.det(A - vals[i] * np.eye(n))}")


def generate_diag_dominant_matrix(
    n: int,
    m: int,
    min_border: int = -100,
    max_border: int = 100,
) -> np.ndarray:
    if n > m:
        raise ValueError("n must be less or equal than m")

    A = np.random.uniform(min_border, max_border, size=(n, n))

    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])

        if A[i, i] >= 0:
            A[i, i] = row_sum + np.random.uniform(1, 10)
        else:
            A[i, i] = -(row_sum + np.random.uniform(1, 10))

    if n != m:
        b = np.random.uniform(min_border, max_border, size=(n, np.abs(n - m)))
        M = np.hstack((A, b)).astype(np.double)

    else:
        M = A

    return M


def generate_symmetric_matrix(
    n: int,
    m: int,
    min_border: int = -100,
    max_border: int = 100,
) -> np.ndarray:
    if n > m:
        raise ValueError("n must be less or equal than m")

    A = np.random.uniform(min_border, max_border, size=(n, m))
    A = A @ A.T

    if n != m:
        b = np.random.uniform(min_border, max_border, size=(n, np.abs(n - m)))
        M = np.hstack((A, b.reshape(-1, 1)))

    else:
        M = A

    return M


def generate_symmetric_pos_def_matrix(
    n: int,
    m: int,
    min_border: int = -100,
    max_border: int = 100,
) -> np.ndarray:
    if n > m:
        raise ValueError("n must be less or equal than m")

    A = np.random.uniform(min_border, max_border, size=(n, m))
    A = A @ A.T + n * np.eye(n)

    if n != m:
        b = np.random.uniform(min_border, max_border, size=(n, np.abs(n - m)))

        M = np.hstack((A, b.reshape(-1, 1)))

    else:
        M = A

    return M


def generate_matrix(
    n: int,
    m: int,
    min_border: int = -100,
    max_border: int = 100,
) -> np.ndarray:
    if n > m:
        raise ValueError("n must be less or equal than m")

    A = np.random.uniform(min_border, max_border, size=(n, m))

    if n != m:
        b = np.random.uniform(min_border, max_border, size=(n, np.abs(n - m)))
        M = np.hstack((A, b.reshape(-1, 1)))

    else:
        M = A

    return M
