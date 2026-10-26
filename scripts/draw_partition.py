"""Статическая схема распределения строк для report.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def main():
    colors = ["#2563eb", "#0f766e", "#9333ea", "#c2410c"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ranks = [0, 0, 1, 1, 2, 2, 3]
    for row, rank in enumerate(ranks):
        for col in range(5):
            ax.add_patch(
                Rectangle(
                    (col * 0.55, 6 - row),
                    0.5,
                    0.85,
                    facecolor=colors[rank],
                    alpha=0.85,
                )
            )
        ax.text(
            -0.2, 6 - row + 0.42, f"Строка {row}", ha="right", va="center", fontsize=11
        )
    for rank, (first, count) in enumerate([(0, 2), (2, 2), (4, 2), (6, 1)]):
        y = 6 - first - (count - 1) / 2 + 0.42
        ax.annotate(
            "",
            xy=(4.15, y),
            xytext=(3.05, y),
            arrowprops={"arrowstyle": "->", "color": colors[rank], "lw": 2},
        )
        ax.text(
            4.35,
            y,
            f"Ранг {rank}: {count} {'строка' if count == 1 else 'строки'}\nначало блока: {first}",
            ha="left",
            va="center",
            fontsize=12,
            color=colors[rank],
        )
    ax.text(
        1.35,
        7.3,
        "Матрица A: 7 строк × 5 столбцов",
        ha="center",
        fontsize=13,
        weight="bold",
    )
    ax.text(5.4, 7.3, "Четыре MPI-процесса", ha="center", fontsize=13, weight="bold")
    ax.text(
        3.0,
        -1.0,
        "Каждый процесс получает весь вектор x длины 5 через Bcast",
        ha="center",
        fontsize=12,
    )
    ax.set(xlim=(-1.7, 8.1), ylim=(-1.5, 8.0))
    ax.axis("off")
    fig.tight_layout()
    target = Path(__file__).resolve().parents[1] / "report" / "partition.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
