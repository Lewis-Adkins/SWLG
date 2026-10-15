
from pathlib import Path

import pandas as pd


class ModelRun:
    def __init__(self, model: object, config: dict):
        self.model  = model
        self.config = config
        self.train_time = 0

    def get_data(self, data_index: int)->pd.DataFrame:
        if data_index == 0:
            self.data = pd.read_csv("data/data.csv")
        else:
            self.data = pd.read_csv(f"data/data_{data_index}.csv")
        return self.data

    def create_results_path(self, dataset_id: int, prediction_time: int) -> str:
        phases_tag   = "phases" if self.config["data"]["use_phases"] else "nophases"
        results_path = Path(f"results/{self.model.name}/{phases_tag}/dataset_{dataset_id}/t+{prediction_time}")
        results_path.mkdir(parents=True, exist_ok=True)
        return str(results_path)

    def pair_input_output(self)->tuple:
        return self.model.pair_input_output()

    def run_model(self,data: pd.DataFrame, prediction_time:float)->None:

            self.model.run()
