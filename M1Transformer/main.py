import os
import numpy as np
import pandas as pd
import argparse
import torch

from torres.load_data import load_series
from torres.m1 import pair_input_output
from torres.m1 import evaluate
from torres.time_series_classification import score_forecast
from torres.stats import tss_f1

from linear.linear_regression import run_linear_baseline

from utils.models import load_config
from utils.file import ensure_dir, _result_base, create_result_dirs, load_checkpoint_safely
from utils.output import create_result_csv, create_f1_csv
from utils.training import train_models, test_model, set_up_models_train_test


def main():
    
    cfg = load_config()
    is_linear = cfg["model_type"] == "linear"
    torch._dynamo.config.recompile_limit = cfg["n_seeds"] * cfg["n_datasets"] * len(cfg["prediction_time"]) + 16
    tag = cfg["run_tag"]
    base = _result_base(cfg)

    if cfg["plot_only"]:
        # Skip data load, training, testing, and evaluation entirely --
        # create_result_csv only reads results.txt files already on disk.
        # f1.csv is left untouched (score_forecast isn't re-run here) since
        # this path doesn't rebuild f1_records. Plotting is a separate manual
        # step now (utils/plot_dataset_comparison.py), not run automatically
        # here.
        create_result_csv(cfg)
        return

    data = pd.read_csv('data/data.csv')

    create_result_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('data/event_timestamps.txt', 'r') as event_file:
        lines = event_file.readlines()
    event_times = [line.split() for line in lines]

    # Fixed per-seed manual_seed base values (kept from the original 5-seed
    # list for continuity); offset by dataset_id below so a given seed index
    # doesn't reuse the exact same torch seed across dataset variants. 
    seed_bases = [1096743781, 1234956713875618956, 1349875190375,
                  236747823658, 149571475189137, 13495671387651,90878906578,
                  495870123985725, 196720470596, 4893574689476, 75829735]

    ### TRAINING ###
    f1_records = []
    tag = cfg["run_tag"]
    for pt in cfg["prediction_time"]:
            trains, targets_trains, tests, targets_tests = pair_input_output(
                data, cfg["use_phases"], int(pt), n_datasets=cfg["n_datasets"], train_split=cfg["train_split"])

            for dataset_id in range(cfg["n_datasets"]):
                print(f"======== Dataset {dataset_id}/{cfg['n_datasets'] - 1}, t+{pt} "
                      f"({len(trains[dataset_id])} train / {len(tests[dataset_id])} test windows) ========")

                if is_linear:
                    # Linear is a peer model_type: it writes into the same
                    # M1-00 result dir a one-seed transformer run would, so
                    # score_forecast / create_result_csv below need no
                    # linear-specific branch. n_seeds is pinned to 1 for it
                    # (utils/models.load_config).
                    result_path = f"results/{base}/resutls_per_dataset/t+{pt}/{tag}/dataset{dataset_id}/M1-00"
                    print(f"======== Fitting linear regression, t+{pt} dataset {dataset_id} ========")
                    run_linear_baseline(trains[dataset_id], targets_trains[dataset_id],
                                        tests[dataset_id], targets_tests[dataset_id],
                                        data, pt, result_path, event_times)
                    if dataset_id == 0:
                        for app in ["app0", "app1", "app2", "app3"]:
                            print(app)
                            f1_records.extend(score_forecast(pt, app, tag, base, n_seeds=cfg["n_seeds"]))
                    continue

                group_size = cfg["models_in_parallel"]
                seed_groups = [range(g, min(g + group_size, cfg["n_seeds"]))
                               for g in range(0, cfg["n_seeds"], group_size)]

                for seeds in seed_groups:
                    models, optimizers, model_names, train_loader, test_loader, loss_fn = set_up_models_train_test(
                        trains[dataset_id], targets_trains[dataset_id],
                        tests[dataset_id], targets_tests[dataset_id], cfg, device, dataset_id, seed_bases, seeds)

                    model_paths = [f"models/{base}/t+{pt}/dataset{dataset_id}/{name}.pt" for name in model_names]

                    if cfg["train_new_models"]:
                        already_done = [os.path.exists(p) for p in model_paths]
                        for name, path, skip in zip(model_names, model_paths, already_done):
                            if skip:
                                print(f"{name} dataset{dataset_id} t+{pt}: checkpoint already exists, skipping training")

                        if not all(already_done):
                            print(f"======== Training {sum(not d for d in already_done)} model(s) together, "
                                  f"t+{pt} dataset {dataset_id}, seeds {list(seeds)} ========")
                            models, _ = train_models(models, optimizers, model_names, train_loader, loss_fn, device,
                                                      n_epoch=cfg["n_epochs"], patience=cfg["patience"],
                                                      min_delta=cfg["min_delta"], already_done=already_done)

                        for model, path, skip in zip(models, model_paths, already_done):
                            if not skip:
                                torch.save(model.state_dict(), path)

                    for i in range(len(seeds)):
                        model_name = model_names[i]
                        model_path = model_paths[i]
                        result_path = f"results/{base}/resutls_per_dataset/t+{pt}/{tag}/dataset{dataset_id}/{model_name}"

                        ### TESTING ###
                        model = load_checkpoint_safely(models[i], model_path)
                        predictions = test_model(model, test_loader, device, loss_fn).detach().cpu().numpy()
                        ensure_dir(result_path)
                        np.savetxt(f"{result_path}/predictions.txt", predictions, delimiter=",")
                        ### FROM TORRES MAIN FUNCTION - EVALUATION ###

                        targets_test = targets_tests[dataset_id]

                        # Only events whose full onset..peak span is present in this
                        # dataset variant's test set -- works for the real dataset
                        # (dataset 0) and any block-partitioned variant alike, since
                        # it checks presence, not position.
                        event_times_test = [event for event in event_times if event[0] in targets_test]

                        ### EVALUTE PREDICTIONS ###
                        target_times = list(targets_test.keys())

                        predictions = {target_times[j]: predictions[j] for j in range(len(predictions))}

                        print(f"======== Evaluating {model_name}, t+{pt} dataset {dataset_id} ========")
                        evaluate(targets_test, predictions, event_times_test, data, path=result_path, display=False)

                if dataset_id == 0:
                    for app in ["app0", "app1", "app2", "app3"]:
                        print(app)
                        f1_records.extend(score_forecast(pt, app, tag, base, n_seeds=cfg["n_seeds"]))



    create_result_csv(cfg)
    create_f1_csv(f1_records, base)

if __name__ == "__main__":
        main()
