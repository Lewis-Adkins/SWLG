"""
Per-timestep flux line plot: actual vs. transformer vs. linear regression,
for every dataset variant, with checkboxes to toggle individual lines.

Usage:
    python utils/plot_dataset_comparison.py
    python utils/plot_dataset_comparison.py --run_tag sin_phases --prediction_time 6

The bar-chart metric comparison (plot_linear_vs_transformer) is commented
out below, not deleted -- uncomment it (and its __main__ call) to get it back.
"""

import os
import sys

# Running this file directly (a plain `python utils/plot_dataset_comparison.py`,
# which is also what an IDE's Run button does) only puts utils/ on sys.path,
# not the repo root -- add it so `from torres...`/`from linear...` resolve
# the same way they do when main.py imports this module normally. Also used
# below to resolve config.yaml/results/data.csv paths so they don't depend
# on the current working directory the script happened to be launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.widgets import CheckButtons

from torres.m1 import pair_input_output
from torres.load_data import load_series

# ---- Bar-chart metric comparison (MAE/PE/lags across dataset variants) --
# commented out per request to only show the per-timestep flux line plot.
# Uncomment this whole block (and its __main__ call below) to get it back.
#
# METRICS = ["Average MAE", "Average PE", "Average O2P lag", "Average O2T lag", "Average ln10 lag"]
# COLOR_LINEAR = "#4C72B0"
# COLOR_TRANSFORMER = "#DD8452"
#
#
# def _plot_metric(ax, merged, metric, show_legend=False):
#     x = np.arange(len(merged))
#     width = 0.35
#
#     ax.bar(x - width / 2, merged[f"{metric}_linear"], width,
#            color=COLOR_LINEAR, label="Linear regression")
#     ax.bar(x + width / 2, merged[f"{metric}_transformer"], width,
#            yerr=merged[f"SE {metric} (seeds)"], capsize=3, ecolor="black", error_kw={"elinewidth": 1},
#            color=COLOR_TRANSFORMER, label="Transformer (seed mean)")
#
#     ax.axhline(0, color="#888888", linewidth=0.8, zorder=0)
#     ax.set_title(metric, fontsize=10)
#     ax.set_xticks(x)
#     ax.set_xticklabels(merged["dataset"], fontsize=8)
#     ax.set_xlabel("dataset", fontsize=8)
#     ax.tick_params(axis="y", labelsize=8)
#     ax.spines[["top", "right"]].set_visible(False)
#     if show_legend:
#         ax.legend(fontsize=8, frameon=False)
#
#
# def plot_linear_vs_transformer(run_tag, results_dir=None, save_dir=None, show=True):
#     """Build one figure per prediction_time present in both CSVs, comparing
#     the linear baseline against the transformer's seed-mean (+/- SE) across
#     every dataset variant. Returns the list of created Figures.
#
#     :param run_tag: e.g. "sin_phases" -- matches main.py's cfg["run_tag"]
#     :param results_dir: base results/ directory; defaults to the repo's own
#         results/ regardless of the current working directory
#     :param save_dir: if given, saves each figure as
#         {save_dir}/{run_tag}_t{pt}_linear_vs_transformer.png
#     :param show: if True, calls plt.show() after building all figures
#     """
#     if results_dir is None:
#         results_dir = os.path.join(REPO_ROOT, "results")
#
#     linear = pd.read_csv(f"{results_dir}/{run_tag}/linear_results.csv")
#     transformer = pd.read_csv(f"{results_dir}/{run_tag}/results_per_dataset.csv")
#
#     common_ts = sorted(set(linear["t"]) & set(transformer["t"]))
#     if not common_ts:
#         raise ValueError(
#             f"No prediction_time is present in both linear_results.csv and "
#             f"results_per_dataset.csv for run_tag='{run_tag}' -- nothing to plot yet."
#         )
#
#     figures = []
#     for pt in common_ts:
#         merged = pd.merge(
#             linear[linear["t"] == pt], transformer[transformer["t"] == pt],
#             on="dataset", suffixes=("_linear", "_transformer"),
#         ).sort_values("dataset")
#
#         fig = plt.figure(figsize=(11, 9))
#         fig.suptitle(f"{run_tag} -- t+{pt}: linear vs. transformer by dataset variant", fontsize=12)
#         gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.55, wspace=0.3)
#
#         ax_mae = fig.add_subplot(gs[0, :])
#         _plot_metric(ax_mae, merged, "Average MAE", show_legend=True)
#
#         remaining = ["Average PE", "Average O2P lag", "Average O2T lag", "Average ln10 lag"]
#         for i, metric in enumerate(remaining):
#             ax = fig.add_subplot(gs[1 + i // 2, i % 2])
#             _plot_metric(ax, merged, metric)
#
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             out_path = f"{save_dir}/{run_tag}_t{pt}_linear_vs_transformer.png"
#             fig.savefig(out_path, dpi=150, bbox_inches="tight")
#             print(f"saved {out_path}")
#
#         figures.append(fig)
#
#     if show:
#         plt.show()
#
#     return figures


def _default_base(config_path=None):
    """Same model_type/split_type/phases_dir derivation as main.py's
    _result_base(), so this script's default paths match the on-disk layout
    without needing them passed in by hand."""
    if config_path is None:
        config_path = os.path.join(REPO_ROOT, "utils", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config["model"]["type"]
    use_phases = config["data"]["use_phases"]
    n_datasets = config["training"]["n_datasets"]
    split_type = "multi-split" if n_datasets > 1 else "single-split"
    phases_dir = "phases" if use_phases else "no-phases"
    return f"{model_type}/{split_type}/{phases_dir}"


def _linear_run_tag(config_path=None):
    """The run_tag main.py's load_config() builds for model_type "linear":
    linear_phases / linear_nophases. Linear is a peer model type now, so its
    predictions live at the standard layout under base "linear/{split}/{phases}"
    -- see _default_linear_base()."""
    if config_path is None:
        config_path = os.path.join(REPO_ROOT, "utils", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    use_phases = config["data"]["use_phases"]
    return f"linear_{'phases' if use_phases else 'nophases'}"


def _default_linear_base(config_path=None):
    """_result_base for model_type "linear": linear/{split_type}/{phases_dir}."""
    if config_path is None:
        config_path = os.path.join(REPO_ROOT, "utils", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    use_phases = config["data"]["use_phases"]
    n_datasets = config["training"]["n_datasets"]
    split_type = "multi-split" if n_datasets > 1 else "single-split"
    phases_dir = "phases" if use_phases else "no-phases"
    return f"linear/{split_type}/{phases_dir}"


def _detect_datasets_and_seeds(results_dir, run_tag, prediction_time, base):
    """Count dataset{j}/ and M1-0X/ subdirectories on disk instead of
    requiring n_datasets/n_seeds to be passed in and kept in sync by hand."""
    root = f"{results_dir}/{base}/resutls_per_dataset/t+{prediction_time}/{run_tag}"
    dataset_dirs = sorted(
        (d for d in os.listdir(root) if d.startswith("dataset")),
        key=lambda d: int(d[len("dataset"):])
    )
    if not dataset_dirs:
        raise ValueError(f"No dataset* directories found under {root}")
    seed_dirs = sorted(d for d in os.listdir(f"{root}/{dataset_dirs[0]}") if d.startswith("M1-"))
    return len(dataset_dirs), len(seed_dirs)


def plot_timestep_lines(run_tag, prediction_time, n_datasets=None, n_seeds=None,
                         data_path=None, results_dir=None, base=None, linear_base=None,
                         train_split=0.8, use_phases=True, show=True, xlim=None):
    """Interactive per-timestep comparison: for every dataset variant, plots
    the actual proton flux, the transformer's seed-averaged prediction, and
    the linear baseline's prediction -- each against that dataset's own
    test-set-relative index (0..n-1), NOT real calendar time, since the
    block-partitioned variants (dataset 1+) have scattered, non-contiguous
    test sets with no shared time axis to plot against.

    Click a checkbox to toggle that exact line; only dataset 0's three
    lines (its actual/transformer/linear) start visible, since all 3 *
    n_datasets lines at once is unreadable.

    n_datasets/n_seeds are auto-detected from results/ if not given.

    :param xlim: optional (start, end) test-set-index range to zoom the
        x-axis to on load -- either side can be None for "auto" on that
        side (e.g. (500, None) zooms in from index 500 onward). Still
        pannable/zoomable further via the normal matplotlib toolbar; this
        just sets where the view starts.
    """
    if data_path is None:
        data_path = os.path.join(REPO_ROOT, "data", "data.csv")
    if results_dir is None:
        results_dir = os.path.join(REPO_ROOT, "results")
    if base is None:
        base = _default_base()
    if linear_base is None:
        linear_base = _default_linear_base()
    linear_run_tag = _linear_run_tag()

    if n_datasets is None or n_seeds is None:
        n_datasets, n_seeds = _detect_datasets_and_seeds(results_dir, run_tag, prediction_time, base)

    data = pd.read_csv(data_path)
    event_path = os.path.join(REPO_ROOT, "data", "event_indices.txt")
    _, _, _, targets_tests = pair_input_output(
        data, use_phases, prediction_time, n_datasets=n_datasets, train_split=train_split,
        event_path=event_path)

    fig, ax = plt.subplots(figsize=(13, 8))
    plt.subplots_adjust(right=0.76)

    cmap = plt.get_cmap("tab10")
    lines = {}
    for j in range(n_datasets):
        color = cmap(j % 10)
        actual_full = np.array(list(targets_tests[j].values()))

        seed_preds = []
        for seed in range(n_seeds):
            model_name = "M1-" + str(seed).zfill(2)
            path = f"{results_dir}/{base}/resutls_per_dataset/t+{prediction_time}/{run_tag}/dataset{j}/{model_name}/predictions.txt"
            seed_preds.append(load_series(path))
        transformer_full = np.mean(np.array(seed_preds), axis=0)

        linear_path = (f"{results_dir}/{linear_base}/resutls_per_dataset/t+{prediction_time}/"
                       f"{linear_run_tag}/dataset{j}/M1-00/predictions.txt")
        linear_full = np.array(load_series(linear_path))

        n = len(actual_full)
        if xlim is not None:
            lo = max(0, xlim[0]) if xlim[0] is not None else 0
            hi = min(n, xlim[1]) if xlim[1] is not None else n
        else:
            lo, hi = 0, n

        x = np.arange(lo, hi)
        actual = actual_full[lo:hi]
        transformer_avg = transformer_full[lo:hi]
        linear_pred = linear_full[lo:hi]

        visible = (j == 0)
        lines[f"dataset{j} actual"], = ax.plot(
            x, actual, color=color, linestyle="-", linewidth=1.2,
            label=f"dataset{j} actual", visible=visible)
        lines[f"dataset{j} transformer"], = ax.plot(
            x, transformer_avg, color=color, linestyle="--", linewidth=1.2,
            label=f"dataset{j} transformer (seed avg)", visible=visible)
        lines[f"dataset{j} linear"], = ax.plot(
            x, linear_pred, color=color, linestyle=":", linewidth=1.2,
            label=f"dataset{j} linear", visible=visible)

    ax.set_xlabel("test-set index (relative to each dataset's own test set, not real time)")
    ax.set_ylabel("ln(flux)")
    ax.set_title(f"{run_tag} -- t+{prediction_time}: actual vs. transformer vs. linear, per dataset variant")
    ax.spines[["top", "right"]].set_visible(False)
    # No explicit set_xlim here: each dataset's arrays are already sliced to
    # [lo, hi) above (clipped to that dataset's own length), so matplotlib's
    # normal autoscale on the actually-plotted data gives the right view on
    # both axes -- setting xlim from the raw, unclipped (start, end) here
    # would show empty space for any dataset shorter than the requested range.

    labels = list(lines.keys())
    visibility = [lines[l].get_visible() for l in labels]
    check_ax = fig.add_axes([0.78, 0.05, 0.21, 0.9])
    check_ax.set_axis_off()
    check = CheckButtons(check_ax, labels, visibility)
    for lbl in check.labels:
        lbl.set_fontsize(7)

    def _toggle(label):
        line = lines[label]
        line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(_toggle)
    fig._checkbuttons = check  # keep a reference so the widget stays interactive

    if show:
        plt.show()

    return fig


def _default_run_tag(config_path=None):
    """Same run_tag derivation main.py's load_config() uses, so hitting Run
    with no arguments plots whatever the current config.yaml points at."""
    if config_path is None:
        config_path = os.path.join(REPO_ROOT, "utils", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config["model"]["type"]
    use_phases = config["data"]["use_phases"]
    return f"{model_type}_{'phases' if use_phases else 'nophases'}"


if __name__ == "__main__":
    # Zoom range asked via input() instead of CLI flags -- just hit Run and
    # answer the prompts, no command-line arguments needed.
    run_tag = _default_run_tag()
    print(f"run_tag: {run_tag}")

    start_in = input("Zoom start index (blank for none): ").strip()
    end_in = input("Zoom end index (blank for none): ").strip()
    start = int(start_in) if start_in else None
    end = int(end_in) if end_in else None
    xlim = (start, end) if (start is not None or end is not None) else None

    plot_timestep_lines(run_tag, 6, xlim=xlim)

    # Bar-chart metric comparison -- uncomment plot_linear_vs_transformer
    # above to bring this back:
    # plot_linear_vs_transformer(run_tag)
