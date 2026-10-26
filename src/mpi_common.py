"""Общие операции ввода и распределения строк для MPI-версий."""

import numpy as np
from mpi4py import MPI

if __package__:
    from .sequential import read_input
else:
    from sequential import read_input  # type: ignore[no-redef]


def auxiliary_arrays_determination(M, numprocs):
    """Количество строк и начальная строка каждого процесса."""
    if M < 0 or numprocs <= 0:
        raise ValueError("M >= 0, numprocs > 0")
    ave, res = divmod(M, numprocs)
    rcounts = np.full(numprocs, ave, dtype=np.int32)
    rcounts[:res] += 1
    displs = np.zeros(numprocs, dtype=np.int32)
    displs[1:] = np.cumsum(rcounts[:-1])
    return rcounts, displs


def prepare(A, x, comm):
    """Рассылка размеров через Bcast; выделение локальной памяти."""
    rank = comm.Get_rank()
    dims = np.zeros(2, dtype=np.int64)
    if rank == 0:
        dims[:] = A.shape
    comm.Bcast(dims, root=0)
    M, N = map(int, dims)
    counts, displs = auxiliary_arrays_determination(M, comm.Get_size())
    local_A = np.empty((int(counts[rank]), N), dtype=np.float64)
    if rank != 0:
        x = np.empty(N, dtype=np.float64)
    return M, N, counts, displs, local_A, x


def run_main(multiply):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    A = x = None
    error = None
    if rank == 0:
        try:
            A, x = read_input()
        except (OSError, ValueError) as exc:
            error = str(exc)
    # Сообщаем об ошибке всем процессам до обменов, чтобы избежать зависания.
    error = comm.bcast(error, root=0)
    if error is not None:
        raise SystemExit(f"Input error: {error}")
    comm.Barrier()
    start = MPI.Wtime()
    b = multiply(A, x, comm)
    elapsed = comm.reduce(MPI.Wtime() - start, op=MPI.MAX, root=0)
    if rank == 0:
        np.savetxt("Results_parallel.dat", b, fmt="%.16e")
        print(f"Execution time: {elapsed:.9f} seconds")
