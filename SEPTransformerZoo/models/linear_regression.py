import pandas as pd
from datetime import datetime
import numpy as np
from sklearn import linear_model
from models.baseline_model import BaselineModel

class LinearRegression(BaselineModel):
    def __init__(self, config:dict):
        super().__init__(config)
        
        self.name = "linearregression"
        self.type = "baseline"


    def run(self, data:pd.DataFrame, prediction_time:float, event_times:any, path:str)-> np.array:
        trains, targets_trains, tests, targets_tests = super().run(data=data, prediction_time=prediction_time)
        X_train = np.array([instance.flatten() for instance in trains])
        X_test = np.array([instance.flatten() for instance in tests])


        start_time = datetime.now()
        lr_model = linear_model.LinearRegression()
        lr_model.fit(X_train, targets_trains)
        y_pred = lr_model.predict(X_test)
        train_time = datetime.now() - start_time


        # print("targets_tests", np.isnan(targets_tests).any())
        # Only events whose full onset..peak span is present in this variant's
        # test set (works for the real dataset and any block-partitioned variant).
        event_times_test = self.test_events(event_times, targets_tests)
        target_times = list(targets_tests.keys())
        predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}


        np.savetxt(f"{path}/predictions.txt", y_pred, fmt='%.7f')
        self.evaluate(targets_tests, predictions, event_times_test, data, path=path, train_time=train_time)

        return y_pred
