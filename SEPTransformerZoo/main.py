"""
Pipeline entry point. Each stage reads what the previous one wrote to disk, so any stage can be rerun alone.

    python main.py setup                    # create missing bootstrap datasets (data_1.csv ...) and the results folder
    python main.py train [--datasets 0-4] [--models sin,tft]
                                            # train every model (or only --models); writes predictions.txt /
                                            # results.txt per run. tft / chronos2 (AutoGluon) need Python 3.12:
                                            # when run with 3.14, main.py starts config admin.python312_path for them
    python main.py analyze                  # bootstrap CSVs + per-event prediction database (after ALL training is done)
    python main.py plot [--show]            # histograms, per-event true vs pred, best-model counts
    python main.py all [--show]             # setup -> train -> analyze -> plot in one go

run_parallel.ps1 runs setup once, then one `train --datasets <group>` window per group; run analyze and
plot after every window has finished.
"""
import argparse
import importlib.util
import subprocess
import traceback
from pathlib import Path

from core.setup import SetUp, AUTOGLUON_MODELS
from core.model_run import ModelRun

from core.analysis import Analysis
from core.analysis import load_prediction_database, run_plots
from core.plotstuff import PlotStuff
from core.analysis import run_results_analysis, build_prediction_database, save_prediction_database

BASELINES = ["linearregression", "posner", "persistence"]
CHAMPIONS = ["sin", "rope", "zero", "rnn", "nn"]
AUTOGLUON = list(AUTOGLUON_MODELS)   # tft, chronos2: champions too, but trained under Python 3.12
ALL_MODELS = BASELINES + CHAMPIONS + AUTOGLUON


# ---------- setup ----------

def stage_setup(setup: SetUp) -> None:
    setup.check_boostrap_data_exists()   # creates any missing data_1.csv ... data_{n-1}.csv
    Path(setup.config["admin"]["results_folder_path"]).mkdir(parents=True, exist_ok=True)


# ---------- train ----------

def run_model_on_dataset(setup: SetUp, model_type: str, ds: int, data) -> None:
    for use_phases in (False, True):
        setup.config["model"]["type"]       = model_type
        setup.config["data"]["use_phases"] = use_phases
        forecast_model = setup.get_model()

        model_run      = ModelRun(model = forecast_model, config = setup.config)
        print(f"{setup.TRAINING_TEXT} {forecast_model.name} | dataset {ds} | phases {use_phases}")

        for pt in setup.config["data"]["prediction_time"]:
            print("-"*2,f"pt {pt}:","-" * 2)
            try:
                forecast_model.run(data = data, prediction_time = pt, event_times = setup.event_timestamps,path = model_run.create_results_path(ds, pt))
            except Exception:
                print(f"{setup.ERROR_TEXT} {model_type} dataset {ds} phases {use_phases} t+{pt} failed, continuing")
                traceback.print_exc()


def parse_datasets(spec: str | None, n_datasets: int) -> list[int]:
    # "0,1,2" or "3-5" runs only those datasets, so several processes can split the work
    # without ever touching the same dataset. Default: all datasets.
    if spec is None:
        return list(range(n_datasets))

    datasets = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            datasets.extend(range(int(start), int(end) + 1))
        else:
            datasets.append(int(part))

    out_of_range = [ds for ds in datasets if not 0 <= ds < n_datasets]
    if out_of_range:
        raise ValueError(f"--datasets {out_of_range} outside 0-{n_datasets - 1} (n_datasets in config.yaml)")
    return datasets


def parse_models(spec: str | None) -> list[str]:
    # "sin,tft" runs only those models. Default: all of them.
    if spec is None:
        return ALL_MODELS
    models = [m.strip().lower() for m in spec.split(",")]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"--models {unknown} unknown; choose from {ALL_MODELS}")
    return models


def run_in_python312(setup: SetUp, models: list[str], dataset_spec: str | None) -> None:
    # AutoGluon can't be installed for Python 3.14, so train these models in a second process on the
    # 3.12 interpreter (config admin.python312_path). Its output goes to this window / log too.
    python312 = setup.config["admin"].get("python312_path")
    if not python312:
        print(f"{setup.ERROR_TEXT} {models} need Python 3.12: set admin.python312_path in config.yaml")
        return
    cmd = [python312, "-u", "main.py", "train", "--models", ",".join(models)]
    if dataset_spec is not None:
        cmd += ["--datasets", dataset_spec]
    print(f"{setup.PROCESS_TEXT} Training {', '.join(models)} with {python312}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"{setup.ERROR_TEXT} {', '.join(models)} training exited with code {result.returncode}")


def stage_train(setup: SetUp, dataset_spec: str | None, model_spec: str | None = None) -> None:
    datasets = parse_datasets(dataset_spec, int(setup.config["training"]["n_datasets"]))
    models   = parse_models(model_spec)
    print(f"{setup.PROCESS_TEXT} Running datasets {datasets}")

    # AutoGluon models run here only if this Python has AutoGluon (the 3.12 .venv); otherwise hand them off
    has_autogluon = importlib.util.find_spec("autogluon") is not None
    elsewhere     = [m for m in models if m in AUTOGLUON and not has_autogluon]

    for model_types in (BASELINES, CHAMPIONS, AUTOGLUON):
        model_types = [m for m in model_types if m in models and m not in elsewhere]
        if not model_types:
            continue
        for ds in datasets:
            print("-"*5,f"dataset {ds}:","-" * 5)
            data = ModelRun(model = setup.model, config = setup.config).get_data(ds)

            for model_type in model_types:
                run_model_on_dataset(setup, model_type, ds, data)

    if elsewhere:
        run_in_python312(setup, elsewhere, dataset_spec)


# ---------- analyze / plot ----------

def make_analysis(setup: SetUp):

    return Analysis(results_folder=setup.config["admin"]["results_folder_path"],
                    train_split=setup.config["data"]["train_split"])


def stage_analyze(setup: SetUp) -> None:
 
    a = make_analysis(setup)
    print(f"{setup.PROCESS_TEXT} Analysing results files...")
    run_results_analysis(a)
    print(f"{setup.PROCESS_TEXT} Building per-event prediction database...")
    save_prediction_database(a, build_prediction_database(a))


def stage_plot(setup: SetUp, show: bool) -> None:

    a = make_analysis(setup)
    print(f"{setup.PROCESS_TEXT} Plotting...")
    run_plots(a, PlotStuff(a, show=show), load_prediction_database(a))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("setup", help="create missing bootstrap datasets and the results folder")
    train = stages.add_parser("train", help="train every model on the given datasets")
    train.add_argument("--datasets", help="e.g. 0-4 or 0,2,5 (default: all)")
    train.add_argument("--models", help=f"e.g. sin,tft (default: all of {','.join(ALL_MODELS)})")
    stages.add_parser("analyze", help="bootstrap CSVs + per-event prediction database")
    plot = stages.add_parser("plot", help="histograms, per-event true vs pred, best-model counts")
    plot.add_argument("--show", action="store_true", help="open the histogram booklet window")
    everything = stages.add_parser("all", help="setup -> train -> analyze -> plot")
    everything.add_argument("--show", action="store_true", help="open the histogram booklet window")
    args = parser.parse_args()

    setup = SetUp()
    if args.stage in ("setup", "train", "all"):
        stage_setup(setup)          # train also makes sure the datasets exist
    if args.stage in ("train", "all"):
        stage_train(setup, getattr(args, "datasets", None), getattr(args, "models", None))
    if args.stage in ("analyze", "all"):
        stage_analyze(setup)
    if args.stage in ("plot", "all"):
        stage_plot(setup, show=args.show)


if __name__ == "__main__":
    main()
