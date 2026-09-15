r"""
Re-plots the paper's two Figure 2 panels (centroid deviation vs. centroid
size N, and CSM's improvement over N=1) from beta_convergence.py's saved
per-backbone CSVs -- no re-embedding / GPU / audio needed. Fig 1 output:
MSE vs N per beta (0.5/1.0/2.0) per backbone. Fig 2 output: grouped bar
chart of MSE at N=1 / CSM / N=100 per backbone, with an arrow from N=1 to CSM.

Usage:
    cd csm
    python speakerlab/bin/plotting/replot_beta_figures.py
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Serif house style to match the paper body (Times). Actual Times New Roman
# isn't installed on Linux -- Liberation Serif is a metric-compatible
# substitute; mathtext (labels like $N$, $\lambda$) uses Computer Modern
# (LaTeX's default) so it reads as serif too.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "cm"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

DEFAULT_CSV_CAMPPLUS = os.path.join(_REPO_ROOT, "figures/beta_convergence/beta_convergence_campplus.csv")
DEFAULT_CSV_ECAPA = os.path.join(_REPO_ROOT, "figures/beta_convergence/beta_convergence_ecapa.csv")
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "figures/beta_convergence")

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 13

BACKBONE_LABELS = {"campplus": "CAM++", "ecapa": "ECAPA-TDNN"}
# Fig 1: color encodes backbone (2 fixed colors); beta is carried by THREE
# redundant cues at once -- linestyle, marker shape, and opacity -- so the
# three beta curves within one color are unambiguous even where they nearly
# overlap (see plot_fig1).
BACKBONE_COLORS = {"campplus": "tab:blue", "ecapa": "tab:orange"}
BETA_LINESTYLES = {2.0: "-", 1.0: "--", 0.5: ":"}
BETA_MARKERS = {2.0: "o", 1.0: "s", 0.5: "^"}
BETA_ALPHAS = {2.0: 0.95, 1.0: 0.75, 0.5: 0.55}
MSE_YLABEL = "Centroid deviation"
# Fig 1's default x-axis is thinned to this subset of the swept N values
# (the full sweep is still used for Fig 2's N=1/N=100 bars) -- fewer points
# keeps the crowded low-N region readable.
DEFAULT_FIG1_N_VALUES = [1,10, 20,30,40, 50, 100]

# Purple ramp, increasing intensity = closer to the oracle (lower MSE):
# N=1 (worst estimate) lightest, CM in the middle, N=100 (best estimate)
# darkest -- the color itself reads as "distance from oracle."
N1_COLOR = "#dadaeb"
CM_COLOR = "#8b3bc0"
N100_COLOR = "#54278f"


def load_beta_csv(csv_path):
    """Returns (n_values, {beta: [mse...]}, mse_cm)."""
    n_values = []
    beta_series = {}
    mse_cm = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        beta_fields = [c for c in reader.fieldnames if c.startswith("mse_beta_")]
        for field in beta_fields:
            beta_series[float(field[len("mse_beta_"):])] = []
        for row in reader:
            n_values.append(int(row["N"]))
            for field in beta_fields:
                beta_series[float(field[len("mse_beta_"):])].append(float(row[field]))
            mse_cm = float(row["mse_cm"])
    return n_values, beta_series, mse_cm


def find_mse_at_n(n_values, mse_list, n_target):
    return mse_list[n_values.index(n_target)]


def plot_fig1(n_values, campp_beta, ecapa_beta, output_path, display_n_values=None):
    """display_n_values: if given, only these N (a subset of n_values) are
    plotted -- e.g. to thin out a crowded x-axis -- while the underlying
    CSVs/n_values keep every swept N."""
    if display_n_values is not None:
        keep_idx = [i for i, n in enumerate(n_values) if n in display_n_values]
        n_values = [n_values[i] for i in keep_idx]
        campp_beta = {beta: [vals[i] for i in keep_idx] for beta, vals in campp_beta.items()}
        ecapa_beta = {beta: [vals[i] for i in keep_idx] for beta, vals in ecapa_beta.items()}

    betas = sorted(set(campp_beta) | set(ecapa_beta), reverse=True)  # 2.0 first -> drawn on top
    zorders = {2.0: 3, 1.0: 2, 0.5: 1}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Backbone-major plotting order -> a single legend(ncol=2) below fills
    # column-major, so column 1 lists CAM++'s three beta lines and column 2
    # lists ECAPA-TDNN's.
    for backbone, beta_series in [("campplus", campp_beta), ("ecapa", ecapa_beta)]:
        color = BACKBONE_COLORS[backbone]
        for beta in betas:
            # No top-level alpha= here (it would fade markers too) -- bake
            # beta's transparency into the LINE color only, so markers stay
            # fully opaque and legible even where curves are faint/overlapping.
            line_color = to_rgba(color, alpha=BETA_ALPHAS[beta])
            ax.plot(
                n_values, beta_series[beta], color=line_color, linewidth=3.0, markersize=6,
                marker=BETA_MARKERS[beta], linestyle=BETA_LINESTYLES[beta], zorder=zorders[beta],
                markerfacecolor=color, markeredgecolor=color,
                label=rf"{BACKBONE_LABELS[backbone]}, $\lambda={beta:g}$",
            )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_xticks(n_values)
    ax.set_xlabel("Centroid size $N$", size=AXIS_LABEL_SIZE)
    ax.set_ylabel(MSE_YLABEL, size=AXIS_LABEL_SIZE)

    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(values, output_path):
    """values: {backbone: (mse_n1, mse_cm, mse_n100)}, mse_n1/mse_n100
    evaluated at --fig2-beta (see main())."""
    backbones = list(values.keys())
    bar_width = 0.26
    group_gap = 0.3
    group_span = 3 * bar_width

    # Must clear each bar's own half-width (bar_width * 0.92 / 2 ~= 0.12) --
    # otherwise the tight xlim below clips the first/last bar's outer edge,
    # which reads as the whole group being crammed against one side.
    pad = bar_width * 0.5

    # Centers the 3 bars within their [x0-pad, x0+group_span+pad] background
    # band (band position/width is unchanged) -- without this the bars sit
    # noticeably left of center in their own band (large empty gap on the
    # right, almost none on the left).
    bar_offset = bar_width / 2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    group_centers = []
    x0 = 0.0
    for backbone in backbones:
        mse_n1, mse_cm, mse_n100 = values[backbone]

        # Faint backbone-colored fill behind the group, no border -- ties
        # this group back to Fig 1's blue/orange backbone encoding without
        # competing with the purple N=1/CM/N=100 bar colors.
        ax.axvspan(x0 - pad, x0 + group_span + pad, color=BACKBONE_COLORS[backbone], alpha=0.08, zorder=0)

        xs = [x0 + bar_offset, x0 + bar_width + bar_offset, x0 + 2 * bar_width + bar_offset]
        bars = [
            (xs[0], mse_n1, N1_COLOR),
            (xs[1], mse_cm, CM_COLOR),
            (xs[2], mse_n100, N100_COLOR),
        ]
        for x, height, color in bars:
            ax.bar(x, height, width=bar_width * 0.92, color=color, edgecolor="white", linewidth=0.8)

        # Arrow from the N=1 bar to the CM bar: visualizes "a single
        # utterance, mapped through the trained estimator, lands here" --
        # i.e. CM replaces N=1's own raw formula estimate, not N=100's.
        ax.annotate(
            "", xy=(xs[1], mse_cm), xytext=(xs[0], mse_n1),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8,
                             shrinkA=6, shrinkB=6, connectionstyle="arc3,rad=0.25"),
        )

        group_centers.append(xs[1])
        x0 += group_span + group_gap

    ax.set_ylim(bottom=0, top=ax.get_ylim()[1])  # headroom so the arrow doesn't touch the top edge
    # Hug the groups tightly (drop matplotlib's default ~5% autoscale margin
    # on both sides) so the whole plot sits left instead of floating with
    # empty space around it.
    last_group_start = x0 - group_span - group_gap
    ax.set_xlim(-pad, last_group_start + group_span + pad)


    handles = [
        plt.Rectangle((0, 0), 1, 1, color=N1_COLOR, label=r"$N=1$"),
        plt.Rectangle((0, 0), 1, 1, color=CM_COLOR, label="CSM"),
        plt.Rectangle((0, 0), 1, 1, color=N100_COLOR, label=r"$N=100$"),
    ]

    ax.set_xticks(group_centers)
    ax.set_xticklabels([BACKBONE_LABELS[b] for b in backbones], size=AXIS_LABEL_SIZE)
    ax.set_ylabel(MSE_YLABEL, size=AXIS_LABEL_SIZE)
    ax.legend(handles=handles, loc="upper right", fontsize=13, ncol=1,)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-campplus", default=DEFAULT_CSV_CAMPPLUS)
    parser.add_argument("--csv-ecapa", default=DEFAULT_CSV_ECAPA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fig1-output", default="beta_sweep_mse.png")
    parser.add_argument("--fig2-output", default="n1_cm_n100_bars.png")
    parser.add_argument("--fig2-beta", type=float, default=2.0,
                         help="Which beta's N=1/N=100 bars Fig 2 draws (must be a column in both CSVs).")
    parser.add_argument("--fig1-n-values", type=int, nargs="+", default=DEFAULT_FIG1_N_VALUES,
                         help="Subset of the swept N values Fig 1 plots.")
    args = parser.parse_args()

    n_campp, campp_beta, mse_cm_campp = load_beta_csv(args.csv_campplus)
    n_ecapa, ecapa_beta, mse_cm_ecapa = load_beta_csv(args.csv_ecapa)
    if n_campp != n_ecapa:
        raise ValueError("--csv-campplus and --csv-ecapa must share the same N sweep for Fig 1's shared x-axis")

    os.makedirs(args.output_dir, exist_ok=True)
    fig1_path = os.path.join(args.output_dir, args.fig1_output)
    fig2_path = os.path.join(args.output_dir, args.fig2_output)

    missing = [n for n in args.fig1_n_values if n not in n_campp]
    if missing:
        raise ValueError(f"--fig1-n-values {missing} not present in the CSVs (available: {n_campp})")
    plot_fig1(n_campp, campp_beta, ecapa_beta, fig1_path, display_n_values=args.fig1_n_values)

    if args.fig2_beta not in campp_beta or args.fig2_beta not in ecapa_beta:
        raise ValueError(f"--fig2-beta {args.fig2_beta:g} is not a column in both CSVs "
                          f"(available: {sorted(set(campp_beta) & set(ecapa_beta))})")
    fig2_values = {
        "campplus": (
            find_mse_at_n(n_campp, campp_beta[args.fig2_beta], 1),
            mse_cm_campp,
            find_mse_at_n(n_campp, campp_beta[args.fig2_beta], 100),
        ),
        "ecapa": (
            find_mse_at_n(n_ecapa, ecapa_beta[args.fig2_beta], 1),
            mse_cm_ecapa,
            find_mse_at_n(n_ecapa, ecapa_beta[args.fig2_beta], 100),
        ),
    }
    plot_fig2(fig2_values, fig2_path)

    print(f"[{fig1_path}]")
    print(f"[{fig2_path}]")


if __name__ == "__main__":
    main()
