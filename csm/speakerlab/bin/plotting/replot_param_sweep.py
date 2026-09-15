r"""
Re-plots param_sweep.py's paper Figure 1 (cos_sim vs alpha, MSE vs beta) from
its already-saved CSV -- no re-embedding / GPU / audio needed. Draws one curve
per backbone found in the CSV (CAM++ blue, ECAPA-TDNN orange); an older
single-backbone CSV (no "backbone" column) is treated as one "campplus" curve.

Usage:
    cd csm
    python speakerlab/bin/plotting/replot_param_sweep.py
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Serif house style to match the paper body (Times). Actual Times New Roman
# isn't installed on Linux -- Liberation Serif is a metric-compatible
# substitute; mathtext (labels like $\alpha$, $\beta$) uses Computer Modern
# (LaTeX's default) so it reads as serif too.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "cm"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

PARAM_SWEEP_CSV = os.path.join(_REPO_ROOT, "figures/param_sweep/param_sweep.csv")
BETA_VS_MSE_OUTPUT = os.path.join(_REPO_ROOT, "figures/param_sweep/beta_vs_mse.png")
ALPHA_VS_COS_SIM_OUTPUT = os.path.join(_REPO_ROOT, "figures/param_sweep/alpha_vs_cos_sim.png")

XLABEL_SIZE = 40
YLABEL_SIZE = 25
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 16

# Same convention as param_sweep.py's own BACKBONE_LINESTYLES/BACKBONE_LABELS/
# BACKBONE_COLORS, and replot_beta_figures.py's shared backbone->color mapping.
BACKBONE_LABELS = {"campplus": "CAM++", "ecapa": "ECAPA-TDNN"}
BACKBONE_LINESTYLES = {"campplus": "-", "ecapa": "-"}
BACKBONE_COLORS = {"campplus": "tab:blue", "ecapa": "tab:orange"}
BACKBONE_ALPHAS = {"campplus": 1.0, "ecapa": 1.0}


def load_param_sweep(csv_path):
    """Returns {backbone: {"alphas": [...], "mean_cos_sim": [...], "betas": [...],
    "mean_mse": [...]}}. Older single-backbone CSVs (no "backbone" column,
    from before param_sweep.py's --backbone both) are treated as one
    implicit "campplus" series."""
    series = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_backbone = "backbone" in reader.fieldnames
        for row in reader:
            backbone = row["backbone"] if has_backbone else "campplus"
            s = series.setdefault(backbone, {"alphas": [], "mean_cos_sim": [], "betas": [], "mean_mse": []})
            s["alphas"].append(float(row["alpha"]))
            s["mean_cos_sim"].append(float(row["mean_cos_sim"]))
            s["betas"].append(float(row["beta"]))
            s["mean_mse"].append(float(row["mean_mse"]))
    return series


def plot_beta_vs_mse(series, output_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for backbone, s in series.items():
        ax.plot(s["betas"], s["mean_mse"], marker="s", color=BACKBONE_COLORS[backbone], linewidth=4.0, markersize=12,
                 linestyle=BACKBONE_LINESTYLES[backbone], alpha=BACKBONE_ALPHAS[backbone],
                 label=BACKBONE_LABELS[backbone])
    ax.axvline(0.5, color="black", linewidth=1.0, linestyle="dotted", alpha=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(next(iter(series.values()))["betas"])
    ax.set_xlabel(r"$\beta$", size=XLABEL_SIZE)
    ax.set_ylabel("MSE", size=YLABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_vs_cos_sim(series, output_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for backbone, s in series.items():
        ax.plot(s["alphas"], s["mean_cos_sim"], marker="s", color=BACKBONE_COLORS[backbone], linewidth=4.0, markersize=12,
                 linestyle=BACKBONE_LINESTYLES[backbone], alpha=BACKBONE_ALPHAS[backbone],
                 label=BACKBONE_LABELS[backbone])
    ax.axvline(0.5, color="black", linewidth=1.0, linestyle="dotted", alpha=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(next(iter(series.values()))["alphas"])
    ax.set_xlabel(r"$\alpha$", size=XLABEL_SIZE)
    ax.set_ylabel("cosine similarity", size=YLABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if len(series) > 1:
        ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--param-sweep-csv", default=PARAM_SWEEP_CSV)
    parser.add_argument("--beta-vs-mse-output", default=BETA_VS_MSE_OUTPUT)
    parser.add_argument("--alpha-vs-cos-sim-output", default=ALPHA_VS_COS_SIM_OUTPUT)
    args = parser.parse_args()

    series = load_param_sweep(args.param_sweep_csv)
    plot_beta_vs_mse(series, args.beta_vs_mse_output)
    plot_alpha_vs_cos_sim(series, args.alpha_vs_cos_sim_output)

    print(f"[{args.beta_vs_mse_output}]")
    print(f"[{args.alpha_vs_cos_sim_output}]")


if __name__ == "__main__":
    main()
