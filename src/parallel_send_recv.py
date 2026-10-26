"""Распределение строк и сбор результата через Send/Recv."""

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
    """Вход задан на процессе 0; результат возвращается только ему."""
    rank = comm.Get_rank()
    M, N, counts, displs, local_A, x = prepare(A, x, comm)
    if rank == 0:
        local_A[:] = A[: counts[0]]
        for dest in range(1, comm.Get_size()):
            first = int(displs[dest])
            last = first + int(counts[dest])
            comm.Send(A[first:last], dest=dest, tag=10)
    else:
        comm.Recv(local_A, source=0, tag=10)
    comm.Bcast(x, root=0)
    local_b = np.dot(local_A, x)
    if rank == 0:
        b = np.empty(M, dtype=np.float64)
        b[: counts[0]] = local_b
        for source in range(1, comm.Get_size()):
            first = int(displs[source])
            last = first + int(counts[source])
            comm.Recv(b[first:last], source=source, tag=20)
        return b
    comm.Send(local_b, dest=0, tag=20)
    return None


def main():
    run_main(parallel_mat_vec_mult)


if __name__ == "__main__":
    main()
