"""Проверки файлов, граничных размеров и настоящих MPI-запусков."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.sequential import read_matrix, read_params, read_vector
from src.sequential import sequential_mat_vec_mult
from src.mpi_common import auxiliary_arrays_determination

ROOT = Path(__file__).resolve().parents[1]


def test_known_answer():
    A = np.arange(1, 16, dtype=float).reshape(5, 3)
    np.testing.assert_array_equal(
        sequential_mat_vec_mult(A, np.array([1.0, 2.0, 3.0])),
        [14.0, 32.0, 50.0, 68.0, 86.0],
    )


@pytest.mark.parametrize("M,p", [(7, 4), (2, 8), (0, 4), (8, 4), (1, 1)])
def test_partition(M, p):
    counts, displs = auxiliary_arrays_determination(M, p)
    assert counts.sum() == M
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(displs, np.r_[0, np.cumsum(counts[:-1])])


def test_readers(tmp_path):
    params = tmp_path / "in.dat"
    params.write_text("3\n2\n")
    assert read_params(params) == (3, 2)
    matrix = tmp_path / "A.dat"
    for text in ("1 2 3\n4 5 6\n", "1\n2\n3\n4\n5\n6\n"):
        matrix.write_text(text)
        np.testing.assert_array_equal(read_matrix(matrix, 2, 3), [[1, 2, 3], [4, 5, 6]])
    vector = tmp_path / "x.dat"
    vector.write_text("2\n")
    assert read_vector(vector).shape == (1,)
    with pytest.raises(ValueError):
        read_matrix(matrix, 3, 3)
    params.write_text("-1\n2\n")
    with pytest.raises(ValueError):
        read_params(params)


@pytest.mark.parametrize(
    "processes,M,N", [(1, 5, 3), (2, 6, 4), (4, 7, 5), (8, 2, 3), (4, 1, 1)]
)
@pytest.mark.parametrize("version", ["parallel_send_recv", "parallel_scatterv"])
def test_mpi(tmp_path, processes, M, N, version):
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    assert launcher, "Нужен MPI; задайте MPIEXEC или добавьте mpiexec в PATH"
    rng = np.random.default_rng(42)
    A, x = rng.normal(size=(M, N)), rng.normal(size=N)
    data = tmp_path / "data"
    data.mkdir()
    (data / "in.dat").write_text(f"{N}\n{M}\n")
    np.savetxt(data / "AData.dat", A)
    np.savetxt(data / "xData.dat", x)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    subprocess.run(
        [
            launcher,
            "-n",
            str(processes),
            sys.executable,
            str(ROOT / "src" / f"{version}.py"),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )
    actual = np.loadtxt(tmp_path / "Results_parallel.dat", ndmin=1)
    np.testing.assert_allclose(actual, A @ x, rtol=1e-12, atol=1e-12)
