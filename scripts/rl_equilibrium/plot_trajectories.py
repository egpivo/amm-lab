"""Paper figure: representative execution trajectories (DQN vs. tuned
lookahead against the oracle path), one standalone panel per episode for
LaTeX subfigures. White background, no in-figure titles (subcaptions carry
win/lose/tie and the seed), paper-sized fonts.

Inputs: out/m3_traj_{dqn,lookahead}_{seed}.csv. Outputs:
out/m3_traj_win.png, out/m3_traj_lose.png, out/m3_traj_tie.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, read_csv

TEXT = "#2B2B2B"
GRID = "#D9D7D1"
ORACLE = "#B8B5AE"
DQN = "#4C6A91"
LOOKAHEAD = "#8A6799"

EPISODES = {
    "win": "30409",
    "lose": "30138",
    "tie": "30266",
}


def load_traj(policy: str, seed: str) -> list[dict]:
    rows = read_csv(OUT / f"m3_traj_{policy}_{seed}.csv")
    return sorted(rows, key=lambda r: int(r["step"]))


def plot_episode(name: str, seed: str, with_legend: bool) -> None:
    dqn = load_traj("dqn", seed)
    la = load_traj("lookahead", seed)

    fig, ax = plt.subplots(figsize=(4.3, 3.7), dpi=200)
    ax2 = ax.twinx()
    ax2.plot(
        [int(r["step"]) for r in la],
        [float(r["oracle_price"]) for r in la],
        color=ORACLE,
        linewidth=1.4,
        zorder=1,
    )
    ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)

    for rows, color, label in ((dqn, DQN, "DQN"), (la, LOOKAHEAD, "Tuned lookahead")):
        ax.plot(
            [int(r["step"]) for r in rows],
            [float(r["remaining_after"]) for r in rows],
            color=color,
            linewidth=2.4,
            drawstyle="steps-post",
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Step", fontsize=18, color=TEXT, labelpad=5)
    ax.set_ylabel("Remaining inventory (Y)", fontsize=18, color=TEXT)
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-2, 52)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", labelsize=16, colors=TEXT)
    if with_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=14, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / f"m3_traj_{name}.png", bbox_inches="tight")
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
        }
    )
    for i, (name, seed) in enumerate(EPISODES.items()):
        plot_episode(name, seed, with_legend=(i == 0))
    print("wrote m3_traj_win.png, m3_traj_lose.png, m3_traj_tie.png")


if __name__ == "__main__":
    main()
