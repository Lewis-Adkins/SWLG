from pathlib import Path
import yaml
import pandas as pd
import sys
import random
import numpy as np
import os

from colorama import Fore, Style

from lib.torres.m1 import pair_input_output

import inspect
from models.linear_regression import LinearRegression
from models.rope import RoPE
from models.sin import Sin
from models.zero import Zero

from models.persistence import Persistence
from models.posner import Posner
from models.nn import NN
from models.rnn import RNN
from models.champion_model import ChampionModel
from torch import nn


class SetUp:
    def __init__(self):

        self.ERROR_TEXT    = Fore.RED    + "[ERROR]"   + Style.RESET_ALL
        self.PROCESS_TEXT  = Fore.YELLOW + "[PROCESS]" + Style.RESET_ALL
        self.TRAINING_TEXT = Fore.BLUE + "[TRAINING]" + Style.RESET_ALL
        try:

            self.config        = self.get_config()
            self.model         = self.get_model()
            self.get_data()
        except FileNotFoundError as fe:
            print(f"{self.ERROR_TEXT} Input files not found, check locations of files {fe}")

    def get_config(self)-> dict:
        with open("config.yaml", "r") as f:
            try:
                print(f"{self.PROCESS_TEXT} Reading config.yaml... ")
                return yaml.safe_load(f)
                
            except yaml.YAMLError as exc:
                print(f"{self.ERROR_TEXT} Reading config.yaml failed: {exc}")

    def check_data_files(self)->None:
        data_folder_path = Path("data")

        if not data_folder_path.is_dir():
            print(f"{self.ERROR_TEXT} 'data' folder does not exist")
            sys.exit()

        data_files = [df.name for df in  data_folder_path.iterdir()]
        file_list_check = ["event_timestamps.txt", "data.csv", "event_indices.txt"]

        for flc in file_list_check:
            if not flc in data_files:
                print(f"{self.ERROR_TEXT} {flc} not found in data folder")
                sys.exit()
        
    def get_event_timestamps(self) -> None:
        print(f"{self.PROCESS_TEXT} Reading event timestamps...")

        with open(self.config["admin"]["data_folder_path"] + "/event_timestamps.txt", 'r') as f:
            lines = f.readlines()

        event_times = [line.split() for line in lines]
        self.event_timestamps = event_times

    def get_particle_data(self) -> None:
        print(f"{self.PROCESS_TEXT} Reading particle data...")
        self.particle_data = pd.read_csv(self.config["admin"]["data_folder_path"]+"/data.csv")

    def get_event_indices(self) -> None:
        print(f"{self.PROCESS_TEXT} Reading event indices...")

        with open(self.config["admin"]["data_folder_path"] + "/event_indices.txt", 'r') as f:
            lines = f.readlines()

        event_indices = [line.split() for line in lines]
        self.event_indices = event_indices

    def get_seeds(self)-> None:
        print(f"{self.PROCESS_TEXT} Getting seeds...")

        with open(self.config["admin"]["data_folder_path"]+"/seeds.txt","r") as f:
            lines = f.read().split()
        self.seeds = [int(seed) for seed in lines]

    def get_model(self)-> object:
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):

                if name.lower() == self.config["model"]["type"]:
                    print(f"{self.PROCESS_TEXT} Getting {name} forecasting model...")
                    if issubclass(obj, nn.Module):
                        return ChampionModel(config=self.config, network_cls=obj)
                    return obj(config=self.config)
                
    def store_model_types(self)-> dict[str, str]:
        model_types = {}
        baselines = []
        champions  =[] 
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if obj.type == "baseline":
                    baselines.append(name)
                else:
                    champions.append(name)

        model_types["baseline"] = baselines
        model_types["champion"] = champions
        return model_types
     
    def get_data(self)-> None:
        self.check_data_files()
        self.get_event_indices()
        self.get_event_timestamps()
        self.get_particle_data()
        self.get_seeds()


    def get_bootstrap_seed(self, bootstrap_name: str) -> int:
        bootstrap_number = int(bootstrap_name.split("_")[1])
        if bootstrap_number > len(self.seeds):
            print(f"{self.ERROR_TEXT} {bootstrap_name} needs seed {bootstrap_number} but seeds.txt only has {len(self.seeds)}")
            sys.exit()
        return self.seeds[bootstrap_number - 1]

    def block_bootstrap_particle_data(self, path: Path, bootstrap_names: list[str])-> tuple[np.array, np.array]:
        particle_data_pd = self.particle_data.reset_index()
        particle_indices = particle_data_pd.index.to_numpy()
        particle_data_np = particle_data_pd.to_numpy()
        
        events = []
        non_background_indicies = np.array([])

        prev_end = 0
        for ei in self.event_indices:
            event_start  = max(int(ei[0]) - 36, prev_end) # don't pad into the previous event
            event_end    = int(ei[3])
            prev_end     = event_end

            single_event_indicies   = np.arange(event_start, event_end)
            non_background_indicies = np.append(non_background_indicies, single_event_indicies)
            particle_event_indicies = np.nonzero(np.isin(particle_indices, single_event_indicies))
            
            event_data              = particle_data_np[particle_event_indicies]

            events.append(event_data)

        particle_background_indicies = np.nonzero(~np.isin(particle_indices, non_background_indicies))
        background                   = particle_data_np[particle_background_indicies]

        n_blocks = int((background.shape[0])**(1/3)) # https://hal.science/hal-03472199/document - optimal block size proportional to n^1/3, n = size of data
        background = np.array_split(background, n_blocks) 
        header = ",".join(particle_data_pd.columns.tolist())
        
        for bn in bootstrap_names:
            blocks = background + events
            random.Random(self.get_bootstrap_seed(bn)).shuffle(blocks)
            bootstrap = np.vstack(blocks)
            np.savetxt(f"{str(path)}/{bn}.csv",bootstrap, fmt = '%s', header = header, comments = "", delimiter=",")

    def check_boostrap_data_exists(self)->dict:

        data_indices = self.config["training"]["n_datasets"] - 1
        data_path = Path(self.config["admin"]["data_folder_path"])
    
        data_files_check = ['data_'+str(i+1) for i in range(data_indices)]
        data_files_present = [df.stem for df in data_path.iterdir()]

        missing_bootstraps = []

        for df in data_files_check:
            if df not in data_files_present:
                print(f"{self.PROCESS_TEXT} Creating {df}.csv bootstrap data")
                missing_bootstraps.append(df)
                

        self.block_bootstrap_particle_data(data_path, missing_bootstraps)
        
        

