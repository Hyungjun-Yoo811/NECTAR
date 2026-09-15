"""
Redraws (or restyles) embedding_analysis_3mix.py's paper Figure 5 t-SNE/PCA
figure from its saved embedding cache (<name>.npz + .json) -- without
re-running CAM++ or re-synthesizing audio, and without torch/torchaudio.
Every visual knob plot_all_triplets exposes (colors, markers, sizes, legend,
triangle outline) is surfaced here as a flag; --palette recolors per triplet.

Usage:
    python speakerlab/bin/plotting/replot_embedding_analysis_3mix.py --input-dir figures/embedding_analysis_3mix/paper_30spk_10triplet
    python speakerlab/bin/plotting/replot_embedding_analysis_3mix.py --palette Set2 --single-marker ^ --mix-marker s --title "3-speaker superposition"
    python speakerlab/bin/plotting/replot_embedding_analysis_3mix.py --no-legend --triangle-linewidth 0 --figsize 8 8
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# _superposition_plot.py lives next to embedding_analysis_3mix.py in
# ../analysis, not in this script's own directory -- reach across to it
# (rather than duplicating plot_all_triplets here) so both scripts always
# draw from the exact same plotting code, with no drift risk.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
_ANALYSIS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "analysis")
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from _superposition_plot import get_palette, plot_all_triplets

DEFAULT_INPUT_DIR = os.path.join(_REPO_ROOT, "figures/embedding_analysis_3mix")


def load_cache(input_dir, cache_name):
    """Loads embedding_analysis_3mix.py's saved cache. Returns (embeddings
    [N, D] ndarray, meta (list of dicts, "color" restored to a tuple from
    JSON's list), triplets (list of (s1, s2, s3) tuples), run_info (seed/n/
    triple_only the cache was generated with))."""
    npz_path = os.path.join(input_dir, cache_name + ".npz")
    json_path = os.path.join(input_dir, cache_name + ".json")
    embeddings = np.load(npz_path)["embeddings"]
    with open(json_path) as f:
        cached = json.load(f)

    meta = cached["meta"]
    for m in meta:
        m["color"] = tuple(m["color"])
    triplets = [tuple(spk_ids) for spk_ids in cached["triplets"]]
    run_info = {k: cached[k] for k in ("seed", "n", "triple_only")}
    return embeddings, meta, triplets, run_info


def recolor(meta, num_triplets, palette_name):
    """Reassigns every point's meta["color"] from its meta["triplet_idx"]
    (NOT from whatever color was cached), via get_palette(num_triplets,
    palette_name) -- palette_name="muted" reproduces the cache's own
    original colors exactly (same algorithm, same order); any other
    matplotlib colormap name gives a different look."""
    colors = get_palette(num_triplets, palette_name)
    for m in meta:
        m["color"] = tuple(colors[m["triplet_idx"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                         help="Directory containing --cache-name's .npz/.json, as saved by embedding_analysis_3mix.py.")
    parser.add_argument("--cache-name", default="superposition_cache")
    parser.add_argument("--output-dir", default=None,
                         help="Defaults to --input-dir (replot in place).")
    parser.add_argument("--output-name", default="superposition_tsne.png")
    parser.add_argument("--pca-output-name", default="superposition_pca_replot.png")
    parser.add_argument("--plot-types", nargs="+", choices=["tsne", "pca"], default=["tsne"])
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None,
                         help="t-SNE/PCA random_state; defaults to the cached run's own --seed.")

    style = parser.add_argument_group("figure style (all optional -- defaults match the original figure)")
    style.add_argument("--palette", default="muted",
                        help="'muted' (default, reproduces the cache's own colors) or any matplotlib colormap name.")
    style.add_argument("--title", default=None, help="Figure title. Omit for no title (the default).")
    style.add_argument("--figsize", type=float, nargs=2, default=[12.0, 10.0], metavar=("WIDTH", "HEIGHT"))
    style.add_argument("--dpi", type=int, default=150)
    style.add_argument("--single-marker", default="*", help="Marker for a single-speaker centroid.")
    style.add_argument("--mix-marker", default="o", help="Marker for a mixture's actual/estimated centroid.")
    style.add_argument("--single-size", type=float, default=460)
    style.add_argument("--mix-centroid-size", type=float, default=520)
    style.add_argument("--mix-estimated-size", type=float, default=520)
    style.add_argument("--estimated-linewidth", type=float, default=3.0,
                        help="Edge thickness of the hollow, dashed estimated-mixture-centroid marker.")
    style.add_argument("--sample-size", type=float, default=45,
                        help="Only visible if the cached run used --all-groups (has sample points).")
    style.add_argument("--legend", dest="show_legend", action="store_true", default=True)
    style.add_argument("--no-legend", dest="show_legend", action="store_false")
    style.add_argument("--legend-loc", default="upper right")
    style.add_argument("--legend-fontsize", type=float, default=14)
    style.add_argument("--triangle-color", default="gray")
    style.add_argument("--triangle-alpha", type=float, default=0.35)
    style.add_argument("--triangle-linewidth", type=float, default=1.0,
                        help="0 hides the per-triplet triangle outline entirely.")
    style.add_argument("--mix-connector-color", default=None,
                        help="Color of the oracle<->estimated connector line; defaults to the triplet color.")
    style.add_argument("--mix-connector-alpha", type=float, default=0.8)
    style.add_argument("--mix-connector-linewidth", type=float, default=4.0,
                        help="0 hides the oracle<->estimated connector line entirely.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    embeddings, meta, triplets, run_info = load_cache(args.input_dir, args.cache_name)
    seed = args.seed if args.seed is not None else run_info["seed"]
    print(f"Loaded {embeddings.shape[0]} cached points ({embeddings.shape[1]}-d) for "
          f"{len(triplets)} triplets from {args.input_dir}/{args.cache_name}.* "
          f"(originally: seed={run_info['seed']}, n={run_info['n']}, triple_only={run_info['triple_only']})")

    recolor(meta, len(triplets), args.palette)

    plot_kwargs = dict(
        title=args.title, figsize=tuple(args.figsize), dpi=args.dpi,
        single_marker=args.single_marker, mix_marker=args.mix_marker,
        single_size=args.single_size, mix_centroid_size=args.mix_centroid_size,
        mix_estimated_size=args.mix_estimated_size, estimated_linewidth=args.estimated_linewidth,
        sample_size=args.sample_size,
        show_legend=args.show_legend, legend_loc=args.legend_loc, legend_fontsize=args.legend_fontsize,
        triangle_color=args.triangle_color, triangle_alpha=args.triangle_alpha,
        triangle_linewidth=args.triangle_linewidth,
        mix_connector_color=args.mix_connector_color, mix_connector_alpha=args.mix_connector_alpha,
        mix_connector_linewidth=args.mix_connector_linewidth,
    )

    if "tsne" in args.plot_types:
        perplexity = min(args.tsne_perplexity, (embeddings.shape[0] - 1) / 3)
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed)
        coords = tsne.fit_transform(embeddings)
        output_path = os.path.join(output_dir, args.output_name)
        plot_all_triplets(coords, meta, triplets, output_path, "t-SNE",
                           "t-SNE dim 1", "t-SNE dim 2", **plot_kwargs)
        print(f"Saved {output_path}")

    if "pca" in args.plot_types:
        pca = PCA(n_components=2, random_state=seed)
        coords_pca = pca.fit_transform(embeddings)
        var1, var2 = pca.explained_variance_ratio_[:2]
        output_path = os.path.join(output_dir, args.pca_output_name)
        plot_all_triplets(coords_pca, meta, triplets, output_path, "PCA",
                           f"PC1 ({var1:.1%} var)", f"PC2 ({var2:.1%} var)", **plot_kwargs)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
