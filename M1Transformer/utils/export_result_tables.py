"""Coalesces every model's per-(seed, dataset) results on disk into one
long-format CSV per statistic -- MAE, PE, O2P lag, O2T lag, ln10 lag, F1, and
training time -- each row identified by (model, t, dataset, seed, ...), with
every requested model.type stacked into the same file. This is a raw export
for pivoting/filtering outside this repo (spreadsheet, stats package, your
own notebook); it deliberately does NOT aggregate anything -- for
mean/SEM-collapsed summaries see utils/output.py's create_result_csv (one
model.type at a time) or utils/plot_bootstrap_comparison.py's plotting
helpers (which consume this same on-disk data but only ever turn it into a
figure, never back into a table).

Reuses utils/plot_bootstrap_comparison.py's path-construction helpers
(_result_base/_run_tag/_default_label), which already generalize over any
model.type string including the autogluon_<name> trees -- "linear" is the
only model with a different, older on-disk layout (no M1-XX seed dirs at
all), handled the same way collect_bar_metrics/collect_f1 already special-
case it.

Usage:
    from utils.export_result_tables import export_result_tables
    from utils.plot_overall_comparison import MODELS

    models = MODELS + ["autogluon_naive", "autogluon_average",
                       "autogluon_temporalfusiontransformer", "autogluon_chronos-2"]
    export_result_tables(models, out_dir="results/combined")

Writes, under out_dir:
    mae.csv, pe.csv, o2p_lag.csv, o2t_lag.csv, ln10_lag.csv  -- columns:
        model, label, t, dataset, seed, value
    f1.csv -- columns:
        model, label, t, dataset, app, alert_window, seed, TP, FN, FP, F1
        (dataset is always 0 -- F1 is only ever scored on the real,
        untouched chronological split, see score_forecast's docstring)
    training_time.csv -- columns:
        model, label, t, dataset, seed, start, end, duration_s
        (rows only exist for (model, t, dataset, seed) combos that actually
        wrote a training_time.txt -- a vmap'd group whose checkpoint already
        existed before a given run never gets one backfilled, see main.py's
        already_done handling, so this can have fewer rows than the metric
        tables for models trained incrementally across multiple sessions)
"""
import glob
import os
import re

import pandas as pd

from utils.plot_bootstrap_comparison import _result_base, _run_tag, _default_label

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# results.txt's "Average X = ..." line prefix -> output CSV file stem
BAR_METRICS = {
    "Average MAE": "mae",
    "Average PE": "pe",
    "Average O2P lag": "o2p_lag",
    "Average O2T lag": "o2t_lag",
    "Average ln10 lag": "ln10_lag",
}

_DATASET_RE = re.compile(r"dataset(\d+)")
_SEED_RE = re.compile(r"M1-(\d+)")


def _parse_dataset_seed(path):
    """Pull dataset_id/seed out of a result path's directory components.
    "linear" has no M1-XX level at all (its one fit per dataset has no seed
    concept) -- seed defaults to 0 in that case, matching M1-00's convention
    everywhere else."""
    parts = path.replace("\\", "/").split("/")
    dataset_id = seed = None
    for p in parts:
        dm = _DATASET_RE.fullmatch(p)
        if dm:
            dataset_id = int(dm.group(1))
        sm = _SEED_RE.fullmatch(p)
        if sm:
            seed = int(sm.group(1))
    return dataset_id, (seed if seed is not None else 0)


def _result_glob(model, pt, n_datasets, use_phases, filename):
    if model == "linear":
        return os.path.join(
            REPO_ROOT, "results", "linear", "multi-split" if n_datasets > 1 else "single-split",
            "phases" if use_phases else "no-phases", f"t+{pt}", "dataset*", filename)
    base = _result_base(model, n_datasets, use_phases)
    tag = _run_tag(model, use_phases)
    return os.path.join(REPO_ROOT, "results", base, "resutls_per_dataset",
                        f"t+{pt}", tag, "dataset*", "M1-*", filename)


def _collect_metric_rows(models, prediction_times, n_datasets, use_phases):
    """{csv_stem: [row, ...]} across every model/pt, parsed from results.txt."""
    out = {stem: [] for stem in BAR_METRICS.values()}
    for model in models:
        label = _default_label(model)
        for pt in prediction_times:
            pattern = _result_glob(model, pt, n_datasets, use_phases, "results.txt")
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"export_result_tables: no results.txt for '{model}' t+{pt}, skipping")
                continue
            for path in files:
                dataset_id, seed = _parse_dataset_seed(path)
                with open(path) as fh:
                    for line in fh:
                        for raw_name, stem in BAR_METRICS.items():
                            if line.startswith(raw_name):
                                value = float(line.split("=")[1])
                                out[stem].append({"model": model, "label": label, "t": pt,
                                                  "dataset": dataset_id, "seed": seed, "value": value})
    return out


def _collect_training_time_rows(models, prediction_times, n_datasets, use_phases):
    rows = []
    for model in models:
        label = _default_label(model)
        for pt in prediction_times:
            pattern = _result_glob(model, pt, n_datasets, use_phases, "training_time.txt")
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"export_result_tables: no training_time.txt for '{model}' t+{pt}, skipping")
                continue
            for path in files:
                dataset_id, seed = _parse_dataset_seed(path)
                start = end = duration = None
                with open(path) as fh:
                    for line in fh:
                        if line.startswith("Start = "):
                            start = line.split("=", 1)[1].strip()
                        elif line.startswith("End = "):
                            end = line.split("=", 1)[1].strip()
                        elif line.startswith("Duration (s) = "):
                            duration = float(line.split("=", 1)[1])
                rows.append({"model": model, "label": label, "t": pt, "dataset": dataset_id,
                            "seed": seed, "start": start, "end": end, "duration_s": duration})
    return rows


def _collect_f1_rows(models, prediction_times, n_datasets, use_phases):
    rows = []
    for model in models:
        label = _default_label(model)
        base = _result_base(model, n_datasets, use_phases)
        path = os.path.join(REPO_ROOT, "results", base, "overall_results", "f1.csv")
        if model == "linear" and not os.path.exists(path):
            # Older cached location -- see plot_overall_comparison.LINEAR_F1_CSV.
            path = os.path.join(REPO_ROOT, "results", "sin", "multi-split", "phases",
                                "overall_results", "linear_f1.csv")
        if not os.path.exists(path):
            print(f"export_result_tables: no f1.csv for '{model}', skipping")
            continue
        df = pd.read_csv(path)
        df = df[df["t"].isin(prediction_times)].copy()
        if df.empty:
            continue
        # f1.csv's own "model" column is actually the SEED name (e.g.
        # "M1-03"), set by score_forecast -- rename before inserting the
        # real model.type, to avoid the collision. linear's cached f1.csv
        # predates that convention and just says "linear" (no seed concept,
        # one fit per dataset) -- defaults to seed 0, matching M1-00 elsewhere.
        df["seed"] = df["model"].str.extract(r"M1-(\d+)")[0].fillna(0).infer_objects(copy=False).astype(int)
        df = df.drop(columns=["model"])
        df.insert(0, "model", model)
        df.insert(1, "label", label)
        df.insert(3, "dataset", 0)  # score_forecast only ever scores dataset 0
        rows.extend(df.to_dict("records"))
    return rows


def export_result_tables(models, out_dir, prediction_times=(6, 12), n_datasets=10, use_phases=True):
    """Write mae.csv, pe.csv, o2p_lag.csv, o2t_lag.csv, ln10_lag.csv, f1.csv,
    and training_time.csv into out_dir, one row per (model, t, dataset, seed)
    [, app, alert_window for f1] combination found on disk for every model in
    `models`. A model with no results yet for a given t is skipped with a
    printed warning, not an error -- same "still mid-run is fine" philosophy
    as plot_bootstrap_comparison.py's plotting functions."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    metric_rows = _collect_metric_rows(models, prediction_times, n_datasets, use_phases)
    for stem, rows in metric_rows.items():
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"{stem}.csv")
        df.to_csv(path, index=False)
        written[stem] = df
        print(f"wrote {path} ({len(df)} rows)")

    f1_rows = _collect_f1_rows(models, prediction_times, n_datasets, use_phases)
    f1_df = pd.DataFrame(f1_rows)
    f1_path = os.path.join(out_dir, "f1.csv")
    f1_df.to_csv(f1_path, index=False)
    written["f1"] = f1_df
    print(f"wrote {f1_path} ({len(f1_df)} rows)")

    tt_rows = _collect_training_time_rows(models, prediction_times, n_datasets, use_phases)
    tt_df = pd.DataFrame(tt_rows)
    tt_path = os.path.join(out_dir, "training_time.csv")
    tt_df.to_csv(tt_path, index=False)
    written["training_time"] = tt_df
    print(f"wrote {tt_path} ({len(tt_df)} rows)")

    return written


if __name__ == "__main__":
    from utils.plot_overall_comparison import MODELS

    autogluon_models = ["autogluon_naive", "autogluon_average",
                        "autogluon_temporalfusiontransformer", "autogluon_chronos-2"]
    export_result_tables(MODELS + autogluon_models, out_dir=os.path.join(REPO_ROOT, "results", "combined"))
