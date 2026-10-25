
from pathlib import Path

import pandas as pd

from lib.utils.paths import results_path as build_results_path


class ModelRun:
    def __init__(self, model: object, config: dict):
        self.model  = model
        self.config = config
        self.train_time = 0

    def get_data(self, data_index: int)->pd.DataFrame:
        self.data = pd.read_csv(f"data/data_{data_index}.csv")
        return self.data

    def create_results_path(self, dataset_id: int, prediction_time: int) -> str:
        phases_tag   = "phases" if self.config["data"]["use_phases"] else "nophases"
        # layout defined in lib/utils/paths.py (Analysis parses it back with parse_results_path)
        results_path = build_results_path(self.config["admin"]["results_folder_path"], self.model.name, phases_tag,
                                          dataset_id, prediction_time)
        results_path.mkdir(parents=True, exist_ok=True)
        return str(results_path)
