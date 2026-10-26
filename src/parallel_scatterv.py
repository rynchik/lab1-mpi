"""Распределение строк и сбор результата через Scatterv/Gatherv."""

import numpy as np
from mpi4py import MPI

if __package__:
    from .mpi_common import auxiliary_arrays_determination, prepare, run_main
    from .sequential import read_params
else:
    from mpi_common import (  # type: ignore[no-redef]
        auxiliary_arrays_determination,
        prepare,
        run_main,
    )
    from sequential import read_params  # type: ignore[no-redef]

__all__ = ["read_params", "auxiliary_arrays_determination", "parallel_mat_vec_mult"]


def parallel_mat_vec_mult(A, x, comm=MPI.COMM_WORLD):
    """Scatterv считает элементы матрицы, Gatherv — элементы результата."""
    rank = comm.Get_rank()
    M, N, counts, displs, local_A, x = prepare(A, x, comm)
    sendbuf = [A, counts * N, displs * N, MPI.DOUBLE] if rank == 0 else None
    comm.Scatterv(sendbuf, local_A, root=0)
    comm.Bcast(x, root=0)
    local_b = np.dot(local_A, x)
    b = np.empty(M, dtype=np.float64) if rank == 0 else None
    recvbuf = [b, counts, displs, MPI.DOUBLE] if rank == 0 else None
    comm.Gatherv(local_b, recvbuf, root=0)
    return b


def main():
    run_main(parallel_mat_vec_mult)


if __name__ == "__main__":
    main()
