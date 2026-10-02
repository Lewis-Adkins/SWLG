"""Three comparison figures pooling every predictions.txt across the FULL
bootstrap (every seed x every dataset variant, both horizons) for one or more
model.types:

  1. plot_prediction_scatter -- predicted vs. true proton flux, one panel per
     model.type, with a dashed y=x reference line for perfect prediction.
  2. plot_overall_bars -- the same style of bar chart as
     results/overall_comparison_t+*.png.
  3. plot_f1_comparison -- the same style of F1-vs-alert-window chart as
     results/f1_comparison_t+*.png.

(2) and (3) are thin figure-building wrappers around
utils/plot_overall_comparison.py's existing collect_bar_metrics/collect_f1/
_err -- those already take a bare model.type string and glob generically
(only "linear" is special-cased, for its older no-seed-dirs layout), so
nothing there needed to change. Only the labeling/coloring layer is new here:
plot_overall_comparison.py hardcodes a fixed MODELS/MODEL_LABELS/MODEL_COLORS
list, so it doesn't know about new model.types (e.g. the autogluon_<name>
trees) without editing that file. These 3 functions take an explicit
`models` list instead, with colors assigned generically (tab20) and labels
either supplied via the optional `labels` dict or derived automatically
(autogluon_deepar -> "DeepAR", by matching forecasting_models.autogluon's
own MODEL_NAMES capitalization; everything else -> title-cased).

A model with no results yet (or only partial results, mid-run) is skipped
with a warning rather than crashing the whole figure -- useful for checking
progress on a run that's still going.

Usage:
    from utils.plot_bootstrap_comparison import (
        plot_prediction_scatter, plot_overall_bars, plot_f1_comparison)

    models = ["rnn", "sin", "autogluon_deepar", "autogluon_dlinear"]
    plot_prediction_scatter(models, horizons=(6, 12), save="results/scatter.png")
    plot_overall_bars(models, pt=6, save="results/bars_t+6.png")
    plot_f1_comparison(models, pt=6, save="results/f1_t+6.png")
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torres.m1 import pair_input_output
from utils.models import NO_TRAIN_TYPES
from utils.plot_overall_comparison import (
    APPROACHES, ALERT_WINDOWS, ALERT_WINDOW_LABELS,
    collect_bar_metrics, collect_f1, _err,
)

# plot_overall_bars colors each box's whiskers/caps/edge by which bootstrap
# group it comes from, instead of printing "(n=...)" on the x-axis:
# NO_TRAIN_TYPES (linear/persistence/posner) are pinned to n_seeds=1 (see
# utils/models.py's load_config), so their boxes pool 1 seed x n_datasets
# runs; everything else (transformers, nn/rnn, every autogluon_<name>) uses
# the full n_seeds x n_datasets bootstrap.
FULL_BOOTSTRAP_COLOR = "#222222"
SINGLE_SEED_COLOR = "#D55E00"


def _is_single_seed(model):
    return model in NO_TRAIN_TYPES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_label(model):
    """model.type -> a readable label when the caller doesn't supply one via
    `labels`. Strips the "autogluon_" prefix and matches it back against
    forecasting_models.autogluon.MODEL_NAMES so e.g. "autogluon_deepar"
    becomes "DeepAR" (its real capitalization) rather than "Deepar";
    everything else just gets title-cased."""
    if model.startswith("autogluon_"):
        zoo = model[len("autogluon_"):]
        try:
            from forecasting_models.autogluon import MODEL_NAMES
            for name in MODEL_NAMES:
                if name.lower() == zoo:
                    return name
        except ImportError:
            pass
        return zoo.replace("_", " ").title()
    return model.replace("_", " ").title()


def _model_colors(models):
    cmap = plt.get_cmap("tab20")
    return {m: cmap((i % 20) / 19) for i, m in enumerate(models)}


def _run_tag(model, use_phases):
    return f"{model}_{'phases' if use_phases else 'nophases'}"


def _result_base(model, n_datasets, use_phases):
    split_type = "multi-split" if n_datasets > 1 else "single-split"
    phases_dir = "phases" if use_phases else "no-phases"
    return f"{model}/{split_type}/{phases_dir}"


def plot_prediction_scatter(models, horizons=(6, 12), dataset_id=0, seed=0, n_datasets=10,
                            train_split=0.8, use_phases=True, storms_only=True, labels=None,
                            max_points_per_model=20000, save=None):
    """Predicted vs. true proton flux for ONE seed (M1-00 by default) of ONE
    dataset variant (dataset_id=0 by default -- the real, untouched
    chronological split, not one of the event-safe block-bootstrap
    variants) -- a single, specific fit per model.type, not pooled across
    seeds.

    `storms_only=True` (default) keeps only points whose true target
    timestamp falls inside one of data/event_timestamps.txt's 39 labeled SEP
    events (onset..end, same catalog evaluate()'s event1.png..eventN.png
    numbering uses) -- dropping the quiet background stretches that
    otherwise dominate the point count and mostly just show every model
    tracking a near-constant baseline (not interesting). Points are colored
    by WHICH storm they belong to (1-indexed, matching that same "Event N"
    numbering), via a discrete/qualitative palette (tab20, cycled if more
    than 20 distinct storms show up) with a real legend -- storm identity is
    categorical, not a continuous quantity, so a colorbar/gradient would
    falsely imply storm 25 is "closer to" storm 26 than storm 35. Only the
    storms actually present in the plotted data get a legend entry. Set
    storms_only=False to include background points too (plotted uncolored,
    since "which storm" doesn't apply to them).

    One ROW per model.type, one COLUMN per horizon (t+6 kept separate from
    t+12, not pooled together) -- so a model whose accuracy differs sharply
    between the two lead times shows that directly instead of blending it
    into one misleading combined panel. Each panel has a dashed y=x
    reference line.

    `max_points_per_model` randomly subsamples a panel's points if there are
    more (rarely triggers now that this is one seed, not pooled).
    """
    labels = labels or {}
    data = pd.read_csv(os.path.join(REPO_ROOT, "data", "data.csv"))
    model_name = "M1-" + str(seed).zfill(2)

    # 1-indexed storm number (matching evaluate()'s "Event N" numbering) for
    # every timestamp that falls inside one of the 39 labeled SEP events'
    # onset..end span; timestamps outside all spans map to None (background).
    # Timestamps are fixed-width ISO strings, so lexicographic comparison
    # already matches chronological order -- no need to parse them.
    with open(os.path.join(REPO_ROOT, "data", "event_timestamps.txt")) as f:
        event_spans = [(line.split()[0], line.split()[3]) for line in f if line.strip()]

    def storm_number(t):
        for i, (onset, end) in enumerate(event_spans, start=1):
            if onset <= t <= end:
                return i
        return None

    # Ground truth depends only on (prediction_time, dataset variant), never
    # on model.type or seed -- computed ONCE per horizon and reused for
    # every model below, not once per (model, horizon). n_datasets/train_split
    # must still match what actually generated the results on disk so
    # dataset_id's split is reconstructed identically; only dataset_id's
    # targets are kept.
    targets_by_pt = {}
    for pt in horizons:
        _, _, _, targets_tests = pair_input_output(
            data, use_phases, int(pt), n_datasets=n_datasets, train_split=train_split)
        targets_by_pt[pt] = targets_tests[dataset_id]

    n_models = len(models)
    n_horizons = len(horizons)
    fig, axes = plt.subplots(n_models, n_horizons, figsize=(4.6 * n_horizons, 4.6 * n_models),
                             squeeze=False, layout="constrained")
    tab20 = plt.get_cmap("tab20").colors
    storms_seen = set()

    panels = []  # (row, col, ax, all_true, all_pred, all_storm, n_points, panel_title)
    for row, model in enumerate(models):
        label = labels.get(model, _default_label(model))
        base = _result_base(model, n_datasets, use_phases)
        run_tag = _run_tag(model, use_phases)

        for col, pt in enumerate(horizons):
            ax = axes[row, col]
            targets_test = targets_by_pt[pt]
            target_times = list(targets_test.keys())
            storm_ids_full = np.array([storm_number(t) for t in target_times], dtype=object)
            storm_mask = storm_ids_full != None if storms_only else None  # noqa: E711 (elementwise, not `is`)
            true_vals_full = np.array(list(targets_test.values()))

            if model == "linear":
                # Older layout, no resutls_per_dataset/M1-XX levels, no M1-XX
                # seed dirs at all (linear has no seeds) -- see
                # forecasting_models/linear_regression.py and
                # utils/plot_overall_comparison.py's collect_bar_metrics,
                # which special-cases the same thing.
                path = os.path.join(REPO_ROOT, "results", "linear", "multi-split"
                                    if n_datasets > 1 else "single-split",
                                    "phases" if use_phases else "no-phases",
                                    f"t+{pt}", f"dataset{dataset_id}", "predictions.txt")
            else:
                path = os.path.join(REPO_ROOT, "results", base, "resutls_per_dataset",
                                    f"t+{pt}", run_tag, f"dataset{dataset_id}", model_name,
                                    "predictions.txt")

            subset_label = "storms only" if storms_only else "storms + background"
            panel_title = f"{label}  t+{pt} ({subset_label})" if col == 0 else f"t+{pt} ({subset_label})"

            if not os.path.exists(path):
                print(f"plot_prediction_scatter: no {model_name} results for '{model}' t+{pt}, skipping")
                ax.set_title(f"{panel_title}\n(no {model_name} results found)", fontsize=10)
                ax.axis("off")
                continue

            preds = np.atleast_1d(np.loadtxt(path, delimiter=","))
            if len(preds) != len(true_vals_full):
                print(f"plot_prediction_scatter: shape mismatch for '{model}' t+{pt}, skipping")
                ax.set_title(f"{panel_title}\n(shape mismatch)", fontsize=10)
                ax.axis("off")
                continue

            if storms_only:
                all_true = true_vals_full[storm_mask]
                all_pred = preds[storm_mask]
                all_storm = storm_ids_full[storm_mask].astype(int)
                storms_seen.update(all_storm.tolist())
            else:
                all_true, all_pred, all_storm = true_vals_full, preds, None

            panels.append((row, col, ax, all_true, all_pred, all_storm, panel_title))

    storm_color = {s: tab20[i % 20] for i, s in enumerate(sorted(storms_seen))}

    for row, col, ax, all_true, all_pred, all_storm, panel_title in panels:
        n_points = len(all_true)

        # MAE/MSE/RMSE and the least-squares fit line are computed on the
        # FULL (true, pred) data for this panel, before any subsampling
        # below -- subsampling is a rendering convenience only and shouldn't
        # affect the reported statistics.
        mae = np.mean(np.abs(all_pred - all_true))
        mse = np.mean((all_pred - all_true) ** 2)
        rmse = np.sqrt(mse)
        slope, intercept = np.polyfit(all_true, all_pred, 1)

        plot_true, plot_pred, plot_storm = all_true, all_pred, all_storm
        if n_points > max_points_per_model:
            rng = np.random.RandomState(0)
            idx = rng.choice(n_points, size=max_points_per_model, replace=False)
            plot_true, plot_pred = all_true[idx], all_pred[idx]
            if plot_storm is not None:
                plot_storm = all_storm[idx]

        if plot_storm is not None:
            ax.scatter(plot_true, plot_pred, c=[storm_color[s] for s in plot_storm],
                      s=6, alpha=0.5, edgecolors="none")
        else:
            ax.scatter(plot_true, plot_pred, s=3, alpha=0.12, color="#4C72B0", edgecolors="none")
        lo = min(all_true.min(), all_pred.min())
        hi = max(all_true.max(), all_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=1.2, label="y = x")
        fit_x = np.array([lo, hi])
        ax.plot(fit_x, slope * fit_x + intercept, "-", color="#C44E52", linewidth=1.6,
               label=f"fit: y={slope:.2f}x{intercept:+.2f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True flux, ln[(cm² s sr MeV)⁻¹]", fontsize=9)
        if col == 0:
            ax.set_ylabel("Predicted flux, ln[(cm² s sr MeV)⁻¹]", fontsize=9)
        ax.set_title(f"{panel_title}\n(n={n_points}, MAE={mae:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f})",
                    fontsize=8.5)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    subset_label = "storms only" if storms_only else "storms + background"
    fig.suptitle(f"Predicted vs. true proton flux -- dataset {dataset_id}, seed {model_name}, "
                f"{subset_label} (t+6 and t+12 shown separately)", fontsize=12)
    if storms_seen:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=storm_color[s],
                              label=f"Storm {s}") for s in sorted(storms_seen)]
        handles.append(plt.Line2D([0], [0], linestyle="--", color="#888888", label="y = x"))
        fig.legend(handles=handles, loc="outside right center", fontsize=8, frameon=False,
                  title="Storm #", ncol=1 if len(storms_seen) <= 20 else 2)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved {save}")
    return fig


def plot_overall_bars(models, pt, labels=None, save=None):
    """Box-and-whisker chart of the overall MAE/PE/O2P-lag/O2T-lag/ln10-lag
    metrics for `models` at horizon t+pt, in the same panel layout as
    results/overall_comparison_t+*.png but showing the full per-(seed,
    dataset) distribution (box = IQR, line = median, whiskers = 1.5*IQR,
    points beyond = outliers) instead of collapsing it to mean ± error bar --
    generalized to take an explicit model list (with optional {model: label}
    overrides) instead of a hardcoded MODELS constant, colored via a generic
    colormap so a new model.type (e.g. autogluon_<name>) never needs a
    manually-added color/label entry to show up."""
    labels = labels or {}
    bar_data = {}
    for m in models:
        try:
            bar_data[m] = collect_bar_metrics(m, pt)
        except FileNotFoundError as e:
            print(f"plot_overall_bars: skipping '{m}' ({e})")
    models = [m for m in models if m in bar_data]
    if not models:
        raise FileNotFoundError(f"No results for any of the requested models at t+{pt}")

    colors = _model_colors(models)
    model_x = np.arange(len(models))

    def _draw(ax, metric):
        values = [bar_data[m][metric] for m in models]
        bp = ax.boxplot(values, positions=model_x, widths=0.6, patch_artist=True,
                        showfliers=True, medianprops={"color": "#222222", "linewidth": 1.4},
                        flierprops={"markersize": 3, "markeredgecolor": "#666666"})
        for i, (patch, m) in enumerate(zip(bp["boxes"], models)):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.7)
            group_color = SINGLE_SEED_COLOR if _is_single_seed(m) else FULL_BOOTSTRAP_COLOR
            patch.set_edgecolor(group_color)
            patch.set_linewidth(1.3)
            for whisker in bp["whiskers"][2 * i:2 * i + 2]:
                whisker.set_color(group_color)
            for cap in bp["caps"][2 * i:2 * i + 2]:
                cap.set_color(group_color)
        ax.axhline(0, color="#888888", linewidth=0.8, zorder=0)
        ax.set_title(metric.replace("Average ", ""), fontsize=10)
        ax.set_xticks(model_x)
        ax.set_xticklabels([labels.get(m, _default_label(m)) for m in models],
                           fontsize=8, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("value (per seed x dataset)", fontsize=8)

    fig = plt.figure(figsize=(max(11, 1.1 * len(models)), 12), layout="constrained")
    fig.suptitle(f"Overall metrics across the bootstrap -- t+{pt} ({pt * 5} min lead)", fontsize=13)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    _draw(fig.add_subplot(gs[0, :]), "Average MAE")
    rest = ["Average PE", "Average O2P lag", "Average O2T lag", "Average ln10 lag"]
    for metric, cell in zip(rest, [gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1]]):
        _draw(fig.add_subplot(cell), metric)

    legend_handles = []
    if any(not _is_single_seed(m) for m in models):
        legend_handles.append(plt.Line2D([0], [0], color=FULL_BOOTSTRAP_COLOR, linewidth=1.6,
                                         label="10 seeds x 10 datasets (n=100)"))
    if any(_is_single_seed(m) for m in models):
        legend_handles.append(plt.Line2D([0], [0], color=SINGLE_SEED_COLOR, linewidth=1.6,
                                         label="1 seed x 10 datasets (n=10)"))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="outside right center", fontsize=9, frameon=False,
                  title="Whisker/edge color")

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved {save}")
    return fig


def plot_f1_comparison(models, pt, err="sem", labels=None, save=None):
    """F1-vs-alert-window chart for `models` at horizon t+pt, one panel per
    alerting approach (W/EW/EAW/AW), in the same style as
    results/f1_comparison_t+*.png -- generalized the same way as
    plot_overall_bars."""
    labels = labels or {}
    f1_data = {}
    for m in models:
        try:
            f1_data[m] = collect_f1(m, pt)
        except FileNotFoundError as e:
            print(f"plot_f1_comparison: skipping '{m}' ({e})")
    models = [m for m in models if m in f1_data]
    if not models:
        raise FileNotFoundError(f"No f1.csv for any of the requested models at t+{pt}")

    colors = _model_colors(models)
    err_label = "SEM" if err == "sem" else "std"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True, layout="constrained")
    axes = axes.flatten()
    fig.suptitle(f"F1 vs. early-warning window across the bootstrap -- t+{pt} "
                f"({pt * 5} min lead)", fontsize=13)
    x = np.arange(len(ALERT_WINDOWS))

    for ax, (app, app_label) in zip(axes, APPROACHES):
        for model in models:
            per_window = f1_data[model]
            means = np.array([np.mean(per_window[(app, aw)])
                              if len(per_window.get((app, aw), [])) else np.nan
                              for aw in ALERT_WINDOWS])
            errs = np.array([_err(per_window.get((app, aw), np.array([])), err)
                             for aw in ALERT_WINDOWS])
            if np.all(np.isnan(means)):
                continue
            ax.errorbar(x, means, yerr=errs, marker="o", markersize=4, capsize=2,
                       linewidth=1.4, color=colors[model],
                       label=labels.get(model, _default_label(model)))
        ax.set_title(f"Approach {app_label}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ALERT_WINDOW_LABELS, fontsize=8)
        ax.set_xlabel("early-warning window", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    for left_ax in (axes[0], axes[2]):
        left_ax.set_ylabel(f"F1  (mean ± {err_label} over seeds, dataset 0)", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved {save}")
    return fig
