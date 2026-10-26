"""Один MPI-запуск: прогрев, повторные измерения, проверка результата."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.parallel_send_recv import parallel_mat_vec_mult as send_recv  # noqa: E402
from src.parallel_scatterv import parallel_mat_vec_mult as scatterv  # noqa: E402
from src.sequential import sequential_mat_vec_mult  # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
A = x = None
row = {"processes": comm.Get_size()}
if rank == 0:
    rng = np.random.default_rng(42)
    A, x = rng.random((200, 100)), rng.random(100)
    for name, fn in [("sequential", sequential_mat_vec_mult), ("numpy", np.dot)]:
        fn(A, x)
        samples = []
        for _ in range(30):
            start = time.perf_counter()
            fn(A, x)
            samples.append(time.perf_counter() - start)
        row[name] = float(np.median(samples))

for name, fn in [("send_recv", send_recv), ("scatterv", scatterv)]:
    samples = []
    for trial in range(35):
        comm.Barrier()
        start = MPI.Wtime()
        result = fn(A, x, comm)
        elapsed = MPI.Wtime() - start
        maximum = comm.reduce(elapsed, op=MPI.MAX, root=0)
        if rank == 0:
            np.testing.assert_allclose(result, A @ x, rtol=1e-12, atol=1e-12)
            if trial >= 5:
                samples.append(maximum)
    if rank == 0:
        row[name] = float(np.median(samples))
        row[name + "_samples"] = samples
if rank == 0:
    print(json.dumps(row))
