import numpy as np
import pandas as pd

from lib.torres.posner_method import pair_input_output as posner_input_output
from lib.torres.m1 import pair_input_output

class ModelRun:
    def __init__(self, model: object, config: dict):
        self.model  = model
        self.config = config

    def get_data(self, data_index: int)->pd.DataFrame:
        if data_index == 0:
            self.data = pd.read_csv("data/data.csv")
        else:
            self.data = pd.read_csv(f"data/bootstrap/data_{data_index}.csv")


    def pair_input_output(self)-> tuple:

        if self.model.name == "posner":
            
            trains, targets_trains, tests, targets_tests = posner_input_output(
                self.data['electron'], self.data['proton'], self.config['data']['prediciton_time']
            )
        else:
            trains, targets_train, test, targets_test = pair_input_output(
                self.data, self.config["data"]["use_phases"], self.config['data']['prediciton_time']
            )
        return trains, targets_trains, tests, targets_tests