"""
Shared t-SNE/PCA scatter-plot rendering for the 3-speaker centroid-space
analysis (paper Figure 5, Sec 5.2): speaker/mixture centroids, samples, and
estimated centroids, colored per triplet. Used by embedding_analysis_3mix.py
and replot_embedding_analysis_3mix.py.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def speaker_colors(n):
    """n distinct, muted qualitative colors: tab20/tab20b/tab20c concatenated
    (60 total), falling back to a desaturated hsv sweep beyond that. Called
    with n=len(triplets) -- color encodes which triplet a point belongs to."""
    palette = (
        [plt.get_cmap("tab20")(i) for i in range(20)]
        + [plt.get_cmap("tab20b")(i) for i in range(20)]
        + [plt.get_cmap("tab20c")(i) for i in range(20)]
    )
    if n <= len(palette):
        return palette[:n]
    return [matplotlib.colors.hsv_to_rgb((i / n, 0.55, 0.75)) for i in range(n)]


def get_palette(n, name="muted"):
    """n distinct per-triplet colors from a named palette -- "muted" (the
    default, see speaker_colors) or any matplotlib colormap name. Qualitative
    colormaps are indexed directly; continuous ones are swept over [0, 1]."""
    if name == "muted":
        return speaker_colors(n)
    cmap = plt.get_cmap(name)
    if hasattr(cmap, "colors"):  # qualitative (ListedColormap)
        k = len(cmap.colors)
        return [cmap(i % k) for i in range(n)]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]  # continuous


def plot_all_triplets(coords, meta, triplets, output_path, method_name="t-SNE",
                       xlabel="t-SNE dim 1", ylabel="t-SNE dim 2", title=None,
                       figsize=(12, 10), dpi=150,
                       single_marker="*", mix_marker="o",
                       single_size=460, mix_centroid_size=520, mix_estimated_size=520,
                       sample_size=45, sample_alpha=0.5,
                       centroid_edgecolor="black", centroid_edgewidth=1.0,
                       estimated_linewidth=3.0, estimated_linestyle="solid",
                       triangle_color="gray", triangle_alpha=0.35, triangle_linewidth=1.0,
                       mix_connector_color=None, mix_connector_alpha=0.8,
                       mix_connector_linewidth=4.0, mix_connector_linestyle="solid",
                       show_legend=True, legend_loc="upper right", legend_fontsize=14):
    """coords: [N, 2] projected points (t-SNE or PCA). meta: length-N list of
    {"kind": sample/centroid/estimated, "role": single/mix, "color": RGBA
    (one per triplet), "label": speaker/group id or None}. triplets: (s1, s2,
    s3) id tuples, used to connect each triplet's single-speaker centroids
    into a light triangle outline. Marker shape encodes role; color encodes
    triplet. Each mixture group's oracle and estimated centroid are also
    connected by a thin line (matched via meta's "label")."""
    fig, ax = plt.subplots(figsize=figsize)

    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault((m["kind"], m["role"], m["color"]), []).append(i)

    single_centroid_xy = {}
    mix_actual_xy, mix_estimated_xy = {}, {}
    for (kind, role, color), idxs in groups.items():
        xy = coords[idxs]
        marker = single_marker if role == "single" else mix_marker
        if kind == "sample":
            ax.scatter(xy[:, 0], xy[:, 1], color=color, marker=marker, s=sample_size,
                       alpha=sample_alpha, edgecolors="none", zorder=2)
        elif kind == "centroid":
            size = single_size if role == "single" else mix_centroid_size
            ax.scatter(xy[:, 0], xy[:, 1], color=color, marker=marker, s=size,
                       edgecolors=centroid_edgecolor, linewidths=centroid_edgewidth, zorder=5)
        else:  # estimated (mix only) -- hollow, dashed edge in the triplet's color
            ax.scatter(xy[:, 0], xy[:, 1], facecolors="none", edgecolors=color, marker=marker,
                       s=mix_estimated_size, linewidths=estimated_linewidth,
                       linestyle=estimated_linestyle, zorder=6)

        # Labels are not drawn on the figure -- used only below to connect
        # each triplet's centroids and each mixture's oracle/estimated pair.
        if kind == "centroid" and role == "single":
            for i in idxs:
                label = meta[i]["label"]
                if label:
                    single_centroid_xy[label] = coords[i]
        elif kind == "centroid" and role == "mix":
            for i in idxs:
                label = meta[i]["label"]
                if label:
                    mix_actual_xy[label] = (coords[i], color)
        elif kind == "estimated":
            for i in idxs:
                label = meta[i]["label"]
                if label:
                    mix_estimated_xy[label] = coords[i]

    # Light triangle frame over each triplet's 3 single-speaker vertices.
    if triangle_linewidth > 0:
        for spk_ids in triplets:
            if all(spk in single_centroid_xy for spk in spk_ids):
                pts = np.array([single_centroid_xy[spk] for spk in spk_ids] + [single_centroid_xy[spk_ids[0]]])
                ax.plot(pts[:, 0], pts[:, 1], color=triangle_color, linewidth=triangle_linewidth,
                        alpha=triangle_alpha, zorder=1)

    # Thin line directly joining each mixture group's oracle <-> estimated centroid.
    if mix_connector_linewidth > 0:
        for label, (actual_xy, color) in mix_actual_xy.items():
            if label in mix_estimated_xy:
                pts = np.array([actual_xy, mix_estimated_xy[label]])
                ax.plot(pts[:, 0], pts[:, 1], color=mix_connector_color or color,
                        linewidth=mix_connector_linewidth, alpha=mix_connector_alpha,
                        linestyle=mix_connector_linestyle, zorder=4)

    if show_legend:
        legend_elems = [
            Line2D([0], [0], marker=single_marker, color="none", markerfacecolor="dimgray",
                   markeredgecolor="black", markersize=16, label="Speaker centroid"),
            Line2D([0], [0], marker=mix_marker, color="none", markerfacecolor="dimgray",
                   markeredgecolor="black", markersize=12, label="Mixture centroid"),
            Line2D([0], [0], marker=mix_marker, color="none", markerfacecolor="none",
                   markeredgecolor="gray", markeredgewidth=1.8,markersize=12, label="Mixture centroid approximation"),
        ]
        if any(m["kind"] == "sample" for m in meta):
            legend_elems.append(
                Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
                       alpha=0.5, markersize=8, label="Individual utterance / mixture sample")
            )
        ax.legend(handles=legend_elems, loc=legend_loc, frameon=True, fontsize=legend_fontsize)

    if title:
        ax.set_title(title)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
