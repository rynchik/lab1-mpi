"""Запустить замеры 200 x 100, сохранить исходные времена, CSV и график."""

import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    if not launcher:
        raise SystemExit("mpiexec не найден; установите MPI или задайте MPIEXEC")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    rows = []
    for p in (1, 2, 4, 8):
        result = subprocess.run(
            [
                launcher,
                "-n",
                str(p),
                sys.executable,
                str(ROOT / "scripts/benchmark_worker.py"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
        print(f"Completed: {p} processes", flush=True)
    report = ROOT / "report"
    report.mkdir(exist_ok=True)
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "shape": [200, 100],
        "seed": 42,
        "warmup": 5,
        "repeats": 30,
        "rows": rows,
    }
    (report / "raw_timings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    baseline = rows[0]["sequential"]
    with (report / "performance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "processes",
                "sequential_s",
                "numpy_s",
                "send_recv_s",
                "scatterv_s",
                "speedup_send_recv",
                "speedup_scatterv",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["processes"],
                    baseline,
                    row["numpy"],
                    row["send_recv"],
                    row["scatterv"],
                    baseline / row["send_recv"],
                    baseline / row["scatterv"],
                ]
            )
    draw_chart(rows, report)


def draw_chart(rows, report):
    """Построить график по замерам без повторного запуска эксперимента."""
    baseline = rows[0]["sequential"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, label in [("send_recv", "Send/Recv"), ("scatterv", "Scatterv/Gatherv")]:
        p = [r["processes"] for r in rows]
        axes[0].plot(p, [baseline / r[name] for r in rows], "o-", label=label)
        axes[1].plot(p, [r[name] * 1e6 for r in rows], "o-", label=label)
    axes[0].set_ylabel("Ускорение относительно циклов Python")
    axes[1].axhline(
        rows[0]["numpy"] * 1e6, color="gray", linestyle="--", label="NumPy, 1 процесс"
    )
    axes[1].set_ylabel("Время, мкс")
    for ax in axes:
        ax.set_xlabel("Число MPI-процессов")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Умножение матрицы на вектор, 200 × 100; медиана 30 измерений")
    fig.tight_layout()
    fig.savefig(report / "speedup.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
