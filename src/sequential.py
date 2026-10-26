"""Последовательное умножение матрицы на вектор."""

import time
from pathlib import Path

import numpy as np


def data_path(filename):
    """Поддержка запуска из корня проекта и из каталога src."""
    path = Path(filename)
    if not path.exists() and not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def read_params(filename="data/in.dat"):
    """Вернуть (N, M): сначала столбцы, затем строки."""
    values = data_path(filename).read_text(encoding="utf-8").split()
    if len(values) != 2:
        raise ValueError("in.dat должен содержать ровно два числа: N и M")
    N, M = map(int, values)
    if N <= 0 or M <= 0:
        raise ValueError("Размеры матрицы должны быть положительными")
    return N, M


def read_matrix(filename="data/AData.dat", M=None, N=None):
    """Читать строки матрицы или по одному элементу на строке."""
    if M is None or N is None:
        raise ValueError("Необходимо задать M и N")
    values = np.loadtxt(data_path(filename), dtype=np.float64).reshape(-1)
    if values.size != M * N:
        raise ValueError("Количество элементов матрицы не равно M * N")
    return values.reshape(M, N)


def read_vector(filename="data/xData.dat"):
    """Читать вектор, сохраняя размерность и для одного элемента."""
    return np.loadtxt(data_path(filename), dtype=np.float64).reshape(-1)


def read_input():
    N, M = read_params()
    A = read_matrix(M=M, N=N)
    x = read_vector()
    if x.size != N:
        raise ValueError("Длина вектора не равна N")
    return A, x


def sequential_mat_vec_mult(A, x):
    """Вычислить каждую строку двумя вложенными циклами."""
    M, N = A.shape
    if x.shape != (N,):
        raise ValueError("Длина вектора не соответствует матрице")
    b = np.zeros(M, dtype=np.float64)
    for j in range(M):
        for i in range(N):
            b[j] += A[j, i] * x[i]
    return b


def main():
    A, x = read_input()
    start = time.perf_counter()
    b = sequential_mat_vec_mult(A, x)
    elapsed = time.perf_counter() - start
    np.savetxt("Results_sequential.dat", b, fmt="%.16e")
    print(f"Execution time: {elapsed:.9f} seconds")


if __name__ == "__main__":
    main()
