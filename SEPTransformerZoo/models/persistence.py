import os
import numpy as np
import pandas as pd
from datetime import datetime
from models.baseline_model import BaselineModel

class Persistence(BaselineModel):
    def __init__(self, config:dict):
        super().__init__(config)
        self.name = "persistence"
        self.type = "baseline"

    def run(self, data:pd.DataFrame, prediction_time:float, event_times:any, path:str)-> np.array:
        _, _, tests, targets_tests = super().run(data=data, prediction_time=prediction_time)

        # Windows are [electron, electron_high, proton, (phases)] x proton[t - 24: t + 1],
        # so [:, 2, -1] is the current proton flux at t
        PROTON_CHANNEL = 2

        start_time = datetime.now()
        y_pred = tests[:, PROTON_CHANNEL, -1]
        train_time = datetime.now() - start_time

        # Only events whose full onset..peak span is present in this variant's
        # test set (works for the real dataset and any block-partitioned variant).
        event_times_test = self.test_events(event_times, targets_tests)
        target_times = list(targets_tests.keys())
        predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}

        np.savetxt(f"{path}/predictions.txt", y_pred, fmt='%.7f')
        self.evaluate(targets_tests, predictions, event_times_test, data, path=path, train_time=train_time)

        return y_pred
