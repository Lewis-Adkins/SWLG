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
from models.lewis_transformers.baseline.linear_regression import LinRegress
from models.lewis_transformers.champion.rope import RoPE
from models.lewis_transformers.champion.sin import Sin
from models.lewis_transformers.champion.zero import Zero

from models.torres_nns.baseline.persistence import persistence
from models.torres_nns.baseline.posner import Posner
from models.torres_nns.champion.nn import NN
from models.torres_nns.champion.rnn import RNN


class SetUp:
    def __init__(self):

        self.ERROR_TEXT    = Fore.RED    + "[ERROR]"   + Style.RESET_ALL
        self.PROCESS_TEXT  = Fore.YELLOW + "[PROCESS]" + Style.RESET_ALL
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
        self.seeds = lines

    def get_model(self)-> object:
        models_folder = Path("models")

        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if name.lower() == self.config["model"]["type"]:
                    print(f"{self.PROCESS_TEXT} Getting {name} forecasting model...")
                    return obj
    def get_data(self)-> None:
        self.check_data_files()
        self.get_event_indices()
        self.get_event_timestamps()
        self.get_particle_data()


    def block_bootstrap_particle_data(self, path: Path, bootstrap_names: list[str])-> tuple[np.array, np.array]:
        particle_data_pd = self.particle_data.reset_index()
        particle_indices = particle_data_pd.index.to_numpy()
        particle_data_np = particle_data_pd.to_numpy()
        
        events = []
        non_background_indicies = np.array([])

        for ei in self.event_indices:
            event_start  = int(ei[0])
            event_end    = int(ei[3])

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
            random.shuffle(blocks)
            bootstrap = np.vstack(blocks)
            np.savetxt(f"{str(path)}/{bn}.csv",bootstrap, fmt = '%s', header = header)

    def check_boostrap_data_exists(self)->dict:

        data_indices = self.config["training"]["n_datasets"] - 1
        data_path = Path(self.config["admin"]["data_folder_path"])
        bootstrap_path = data_path / "bootstraps"

        if not bootstrap_path.exists():
            bootstrap_path.mkdir()
        else:

            data_files_check = ['data_'+str(i+1) for i in range(data_indices)]
            data_files_present = [df.stem for df in bootstrap_path.iterdir()]

            missing_bootstraps = []

            for df in data_files_check:
                if df not in data_files_present:
                    print(f"{self.PROCESS_TEXT} Creating {df}.csv bootstrap data")
                    missing_bootstraps.append(df)
                

            self.block_bootstrap_particle_data(bootstrap_path, missing_bootstraps)
        
        

