# from core.forecasting_model_run import ModelRunSetUp
from core.setup import SetUp
from core.model_run import ModelRun

import traceback
import numpy as np

BASELINES = []
CHAMPIONS = ["nn", "rnn", "sin", "rope", "zero"]


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


def main():
    setup = SetUp()
    setup.check_boostrap_data_exists()

    for model_types in (BASELINES, CHAMPIONS):
        for ds in range(int(setup.config["training"]["n_datasets"])):
            print("-"*5,f"dataset {ds}:","-" * 5)
            data = ModelRun(model = setup.model, config = setup.config).get_data(ds)

            for model_type in model_types:
                run_model_on_dataset(setup, model_type, ds, data)

if __name__ == "__main__":
    main()
