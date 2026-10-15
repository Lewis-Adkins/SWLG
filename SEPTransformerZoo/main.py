# from core.forecasting_model_run import ModelRunSetUp
from core.setup import SetUp
from core.model_run import ModelRun

import numpy as np



def main():
    setup = SetUp()
    setup.check_boostrap_data_exists()

    forecast_model = setup.model

    model_run = ModelRun(model = forecast_model, config = setup.config)
    print(f"{setup.TRAINING_TEXT} {forecast_model.name}:")
    for ds in range(int(setup.config["training"]["n_datasets"])):
        print("-"*5,f"dataset {ds}:","-" * 5)

        data = model_run.get_data(ds)

        for pt in setup.config["data"]["prediction_time"]:
            print("-"*2,f"pt {pt}:","-" * 2)

            predictions = forecast_model.run(data = data, prediction_time = pt, event_times = setup.event_timestamps,path = model_run.create_results_path(ds, pt))

if __name__ == "__main__":
    main()