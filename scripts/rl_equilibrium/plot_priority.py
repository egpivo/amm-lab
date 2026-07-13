"""Paper figure: priority retraining heatmaps, one standalone panel per
subfigure. White background, no in-figure titles (subcaptions carry them),
paper-sized fonts.

Inputs: out/m3r_priority.csv (DQN train x eval cells, test seed set),
out/m3r_reference.csv (lookahead per evaluation ordering). Outputs:
out/m3r_priority_abs.png, out/m3r_priority_edge.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, mean, read_csv

TEXT = "#2B2B2B"
ORDERS = ["before", "random", "after"]
TICK_LABELS = ["agent-first", "randomized", "agent-last"]


def load_grids() -> tuple[list[list[float]], list[list[float]]]:
    priority = read_csv(OUT / "m3r_priority.csv")
    reference = read_csv(OUT / "m3r_reference.csv")

    lookahead = {}
    for order in ORDERS:
        rows = [
            float(r["shortfall_bps"])
            for r in reference
            if r["policy"] == "lookahead"
            and r["agent_order"] == order
            and r["mode"] == "dynamic_duopoly"
            and r["seed_set"] == "test"
        ]
        lookahead[order] = mean(rows)

    absolute, edge = [], []
    for train in ORDERS:
        abs_row, edge_row = [], []
        for test in ORDERS:
            cells = [
                float(r["shortfall_bps"])
                for r in priority
                if r["train_order"] == train
                and r["test_order"] == test
                and r["seed_set"] == "test"
            ]
            value = mean(cells)
            abs_row.append(value)
            edge_row.append(value - lookahead[test])
        absolute.append(abs_row)
        edge.append(edge_row)
    return absolute, edge


def draw(grid: list[list[float]], out_name: str, cmap: str,
         signed: bool, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=200)
    if signed:
        vmax = max(abs(v) for row in grid for v in row)
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(grid, cmap=cmap)
    for i in range(3):
        for j in range(3):
            value = grid[i][j]
            text = f"{value:+.1f}" if signed else f"{value:.1f}"
            r, g, b, _ = im.cmap(im.norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, text, ha="center", va="center", fontsize=18,
                    color="white" if luminance < 0.5 else TEXT)
    ax.set_xticks(range(3), TICK_LABELS, fontsize=15)
    ax.set_yticks(range(3), TICK_LABELS, fontsize=15)
    ax.set_xlabel("Evaluation ordering", fontsize=16, labelpad=8)
    ax.set_ylabel("Training ordering", fontsize=16, labelpad=8)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Source Sans 3",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    absolute, edge = load_grids()
    draw(absolute, "m3r_priority_abs.png", "Blues", signed=False,
         cbar_label="Shortfall (bps)")
    draw(edge, "m3r_priority_edge.png", "RdBu_r", signed=True,
         cbar_label="DQN $-$ lookahead (bps)")
    print("wrote m3r_priority_abs.png, m3r_priority_edge.png")


if __name__ == "__main__":
    main()
