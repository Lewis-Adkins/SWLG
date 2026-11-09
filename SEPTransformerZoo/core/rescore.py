"""
Re-score saved runs without retraining: rewrite every results.txt from its predictions.txt, e.g. after adding a
metric to BaselineModel.evaluate. Each run keeps its own Train Time and Parameters lines (training isn't repeated),
and every old results.txt is copied to results_backup/<timestamp>/ first.

    python main.py rescore                 # every model
    python main.py rescore --models nn,sin # only these
"""
import contextlib
import io
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lib.utils.events import n_test_samples
from lib.utils.paths import parse_results_path
from models.baseline_model import BaselineModel


def test_targets(dataset: pd.DataFrame, prediction_time: int, train_split: float) -> dict[str, float]:
    # {timestamp: proton} of the test targets, the same ones BaselineModel.pair_input_output returns as targets_test:
    # one target per t in range(24, n - prediction_time) at row t + prediction_time, test = the last (1 - split).
    n_test = n_test_samples(len(dataset), prediction_time, train_split)
    test_rows = dataset.iloc[len(dataset) - n_test:]
    return dict(zip(test_rows['time'], test_rows['proton'].astype(float)))


def kept_lines(results_text: str) -> tuple[str | None, str | None]:
    # The run's Train Time value and its Parameters line, which re-scoring must not change
    train_time, parameters_line = None, None
    for line in results_text.splitlines():
        if line.startswith('Train Time ='):
            train_time = line.split('=', 1)[1].strip()
        elif line.startswith('Parameters ='):
            parameters_line = line
    return train_time, parameters_line


def rescore_results(config: dict, event_timestamps: list, model_names: list[str]) -> None:
    results_folder = Path(config["admin"]["results_folder_path"])
    data_folder    = Path(config["admin"]["data_folder_path"])
    train_split    = config["data"]["train_split"]
    backup_folder  = Path("results_backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
    scorer         = BaselineModel(config)

    # Every run folder with saved predictions, grouped by dataset so each data_<N>.csv is read once
    prediction_files = [path for model_name in model_names if (results_folder / model_name).is_dir()
                        for path in (results_folder / model_name).rglob("predictions.txt")]
    runs_by_dataset = {}
    for prediction_file in prediction_files:
        run_info = parse_results_path(results_folder, prediction_file)
        runs_by_dataset.setdefault(run_info['dataset'], []).append((prediction_file, run_info))
    print(f"Re-scoring {len(prediction_files)} runs; old results.txt files -> {backup_folder}/")

    n_rescored = 0
    for dataset_name, runs in sorted(runs_by_dataset.items()):
        dataset = pd.read_csv(data_folder / f"data_{dataset_name.split('_')[1]}.csv")
        targets_by_horizon = {}
        for prediction_file, run_info in runs:
            run_folder   = prediction_file.parent
            results_file = run_folder / "results.txt"
            prediction_time = int(run_info['t'].split('+')[1])
            if prediction_time not in targets_by_horizon:
                targets_by_horizon[prediction_time] = test_targets(dataset, prediction_time, train_split)
            targets_test = targets_by_horizon[prediction_time]

            predicted_values = np.loadtxt(prediction_file)
            if len(predicted_values) != len(targets_test):
                print(f"  SKIPPED {run_folder}: {len(predicted_values)} predictions for {len(targets_test)} targets")
                continue
            train_time, parameters_line = kept_lines(results_file.read_text()) if results_file.is_file() else (None, None)

            if results_file.is_file():
                backup_file = backup_folder / results_file.relative_to(results_folder)
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(results_file, backup_file)

            predictions = dict(zip(targets_test.keys(), predicted_values))
            events_in_test = scorer.test_events(event_timestamps, targets_test)
            with contextlib.redirect_stdout(io.StringIO()):   # evaluate prints every average; keep the log short
                scorer.evaluate(targets_test, predictions, events_in_test, dataset, str(run_folder), train_time,
                                save_plots=False)
            if parameters_line is not None:
                with open(results_file, "a") as outfile:
                    outfile.write(parameters_line + "\n")
            n_rescored += 1
        print(f"  {dataset_name}: {len(runs)} runs")
    print(f"Re-scored {n_rescored} of {len(prediction_files)} runs")
