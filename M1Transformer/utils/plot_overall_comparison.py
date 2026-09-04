"""
Overall model comparison across the full bootstrap: linear vs. the three
transformer variants (rope / zero / sin).

Per prediction horizon (t+6, t+12) it writes TWO figures:

  * f1_comparison_t+{pt}.png -- F1 vs. early-warning (alert) window, one panel
    per alerting approach (W / EW / EAW / AW), one line per model. F1 is only
    ever scored on dataset 0 (the scoring is positional and invalid for the
    block-partitioned bootstrap variants -- see
    torres/time_series_classification.py and notes/debug), so the error bars
    are the standard error over the 10 training seeds on dataset 0. Linear has
    a single fit, so it's a bare line with no band.

  * overall_comparison_t+{pt}.png -- overall MAE, PE, O2P-lag, O2T-lag and
    ln10-lag bars, one bar per model. The bar is the mean over ALL
    per-(seed, dataset) results -- 100 for each transformer (10 seeds x 10
    bootstrap datasets), 10 for linear (1 fit per dataset, no seeds). The
    error bar is the standard error of that mean, std/sqrt(n) (pass --err std
    for the plain standard deviation instead).

Usage:
    python utils/plot_overall_comparison.py                # build all, show
    python utils/plot_overall_comparison.py --save         # also write PNGs
    python utils/plot_overall_comparison.py --err sem --no-show

Reads results/ straight off disk -- run main.py for each model.type first
(rope / zero / sin / linear) so all four result trees exist.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["linear", "rope", "zero", "sin"]
MODEL_LABELS = {"linear": "Linear", "rope": "RoPE", "zero": "Zero-init", "sin": "Sinusoidal"}
MODEL_COLORS = {"linear": "#888888", "rope": "#4C72B0", "zero": "#DD8452", "sin": "#55A868"}

APPROACHES = [("app0", "W"), ("app1", "EW"), ("app2", "EAW"), ("app3", "AW")]
ALERT_WINDOWS = [1, 3, 6, 9, 12, 24, 72]                       # timesteps (5 min each)
ALERT_WINDOW_LABELS = ["5m", "15m", "30m", "45m", "1h", "2h", "6h"]

BAR_METRICS = ["Average MAE", "Average PE", "Average O2P lag",
               "Average O2T lag", "Average ln10 lag"]

SPLIT, PHASES = "multi-split", "phases"
# Where the linear baseline's per-(t, app, window) F1 was cached by the old
# training_linear path. The refactor to model.type: linear no longer writes
# this file, so fall back gracefully if it isn't there.
LINEAR_F1_CSV = os.path.join(
    REPO_ROOT, "results", "sin", SPLIT, PHASES, "overall_results", "linear_f1.csv")


def _phases_tag(model):
    return f"{model}_phases"


def _results_root(model):
    return os.path.join(REPO_ROOT, "results", model, SPLIT, PHASES)


def collect_bar_metrics(model, pt):
    """{metric: np.array of per-run values} for one model at horizon t+pt.
    Transformers: one value per seed per dataset (glob dataset*/M1-*). Linear:
    one value per dataset (old layout, no seed dirs)."""
    if model == "linear":
        pattern = os.path.join(
            REPO_ROOT, "results", "linear", SPLIT, PHASES, f"t+{pt}", "dataset*", "results.txt")
    else:
        pattern = os.path.join(
            _results_root(model), "resutls_per_dataset", f"t+{pt}", _phases_tag(model),
            "dataset*", "M1-*", "results.txt")

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results.txt for model '{model}' t+{pt} (looked at {pattern})")

    out = {m: [] for m in BAR_METRICS}
    for path in files:
        with open(path) as fh:
            for line in fh:
                for m in BAR_METRICS:
                    if line.startswith(m):
                        out[m].append(float(line.split("=")[1]))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def collect_f1(model, pt):
    """{(app, alert_window): np.array of F1 values} for one model at t+pt.
    Transformers: one value per seed (dataset 0 only). Linear: a single value."""
    if model == "linear":
        if not os.path.exists(LINEAR_F1_CSV):
            return {}
        df = pd.read_csv(LINEAR_F1_CSV)
    else:
        df = pd.read_csv(os.path.join(_results_root(model), "overall_results", "f1.csv"))

    df = df[df["t"] == pt]
    return {
        (app, aw): df[(df["app"] == app) & (df["alert_window"] == aw)]["F1"].to_numpy(dtype=float)
        for app, _ in APPROACHES
        for aw in ALERT_WINDOWS
    }


def _err(values, kind):
    if len(values) == 0:
        return np.nan
    s = np.std(values)
    return s / np.sqrt(len(values)) if kind == "sem" else s


def build_f1_figure(pt, err="std"):
    f1_data = {m: collect_f1(m, pt) for m in MODELS}
    err_label = "SEM" if err == "sem" else "std"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True, layout="constrained")
    axes = axes.flatten()
    fig.suptitle(f"F1 vs. early-warning window across the bootstrap  —  t+{pt} "
                 f"({pt * 5} min lead)", fontsize=13)
    x = np.arange(len(ALERT_WINDOWS))

    for ax, (app, app_label) in zip(axes, APPROACHES):
        for model in MODELS:
            per_window = f1_data[model]
            means = np.array([np.mean(per_window[(app, aw)])
                              if len(per_window.get((app, aw), [])) else np.nan
                              for aw in ALERT_WINDOWS])
            errs = np.array([_err(per_window.get((app, aw), np.array([])), err)
                             for aw in ALERT_WINDOWS])
            if np.all(np.isnan(means)):
                continue
            ax.errorbar(x, means, yerr=errs, marker="o", markersize=4, capsize=2,
                        linewidth=1.4, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.set_title(f"Approach {app_label}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ALERT_WINDOW_LABELS, fontsize=8)
        ax.set_xlabel("early-warning window", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    for left_ax in (axes[0], axes[2]):
        left_ax.set_ylabel(f"F1  (mean ± {err_label} over 10 seeds, dataset 0)", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    return fig


def build_bar_figure(pt, err="std"):
    bar_data = {m: collect_bar_metrics(m, pt) for m in MODELS}
    err_label = "SEM" if err == "sem" else "std"
    model_x = np.arange(len(MODELS))

    def _draw(ax, metric):
        means = [np.mean(bar_data[m][metric]) for m in MODELS]
        errs = [_err(bar_data[m][metric], err) for m in MODELS]
        counts = [len(bar_data[m][metric]) for m in MODELS]
        ax.bar(model_x, means, yerr=errs, capsize=4,
               color=[MODEL_COLORS[m] for m in MODELS],
               error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
        ax.axhline(0, color="#888888", linewidth=0.8, zorder=0)
        ax.set_title(metric.replace("Average ", ""), fontsize=10)
        ax.set_xticks(model_x)
        ax.set_xticklabels([f"{MODEL_LABELS[m]}\n(n={c})" for m, c in zip(MODELS, counts)],
                           fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(f"mean ± {err_label}", fontsize=8)

    fig = plt.figure(figsize=(11, 12), layout="constrained")
    fig.suptitle(f"Overall metrics across the bootstrap  —  t+{pt} "
                 f"({pt * 5} min lead)", fontsize=13)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # MAE spans the full width on top, the remaining four fill a 2x2 grid.
    _draw(fig.add_subplot(gs[0, :]), "Average MAE")
    rest = ["Average PE", "Average O2P lag", "Average O2T lag", "Average ln10 lag"]
    for metric, cell in zip(rest, [gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1]]):
        _draw(fig.add_subplot(cell), metric)

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--err", choices=["std", "sem"], default="sem",
                    help="error bars: standard error of the mean, std/sqrt(n) (default), "
                         "or plain std of the runs")
    ap.add_argument("--save", action="store_true", help="write PNGs to results/")
    ap.add_argument("--horizons", type=int, nargs="+", default=[6, 12])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    for pt in args.horizons:
        f1_fig = build_f1_figure(pt, err=args.err)
        bar_fig = build_bar_figure(pt, err=args.err)
        if args.save:
            for name, fig in [(f"f1_comparison_t+{pt}.png", f1_fig),
                              (f"overall_comparison_t+{pt}.png", bar_fig)]:
                out = os.path.join(REPO_ROOT, "results", name)
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
