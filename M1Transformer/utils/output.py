import os

import numpy as np
import pandas as pd

from utils.file import _result_base


def create_result_csv(cfg):
    """Aggregates transformer MAE/PE/lag metrics two ways: per (t, dataset)
    across seeds (mean + standard error of the mean), and per t across
    dataset variants (mean + std of each dataset's mean) -- the
    dataset-to-dataset spread is the real uncertainty estimate; seed spread
    alone understates it."""
    tag = cfg["run_tag"]
    base = _result_base(cfg)
    metric_names = ["Average MAE", "Average PE", "Average O2P lag",
                     "Average O2T lag", "Average ln10 lag"]
    n_seeds = cfg["n_seeds"]
    n_datasets = cfg["n_datasets"]

    per_dataset_rows = []
    dataset_means = {pt: {m: [] for m in metric_names} for pt in cfg["prediction_time"]}

    for pt in cfg["prediction_time"]:
        for dataset_id in range(n_datasets):
            metrics = {m: [] for m in metric_names}
            for seed in range(n_seeds):
                model_name = "M1-" + str(seed).zfill(2)
                path = f"results/{base}/resutls_per_dataset/t+{pt}/{tag}/dataset{dataset_id}/{model_name}/results.txt"
                with open(path) as file:
                    for line in file:
                        for m in metric_names:
                            if line.startswith(m):
                                metrics[m].append(float(line.split("=")[1].replace(" ", "")))

            row = {"t": pt, "dataset": dataset_id}
            for m in metric_names:
                seed_mean = np.mean(metrics[m])
                seed_se = np.std(metrics[m]) / np.sqrt(n_seeds)
                row[m] = round(seed_mean, 3)
                row[f"SE {m} (seeds)"] = round(seed_se, 3)
                dataset_means[pt][m].append(seed_mean)
            per_dataset_rows.append(row)

    per_dataset_df = pd.DataFrame(per_dataset_rows)

    summary_rows = []
    for pt in cfg["prediction_time"]:
        row = {"t": pt}
        for m in metric_names:
            means = dataset_means[pt][m]
            row[f"mean {m}"] = round(np.mean(means), 3)
            row[f"std {m} (across datasets)"] = round(np.std(means), 3)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)



    os.makedirs(f"results/{base}/overall_results", exist_ok=True)
    per_dataset_df.to_csv(f"results/{base}/overall_results/results_per_dataset.csv", index=False)
    summary_df.to_csv(f"results/{base}/overall_results/results.csv", index=False)
    return per_dataset_df, summary_df

def create_f1_csv(f1_records, base):
    """Save the TP/FN/FP/F1 records gathered from score_forecast (per t,
    app, alert_window, model) to results/{base}/overall_results/f1.csv."""
    os.makedirs(f"results/{base}/overall_results", exist_ok=True)
    f1_df = pd.DataFrame(f1_records)
    f1_df.to_csv(f"results/{base}/overall_results/f1.csv", index=False)
    return f1_df

