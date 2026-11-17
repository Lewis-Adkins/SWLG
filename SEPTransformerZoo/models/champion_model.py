import inspect
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from colorama import Fore, Style
from torch import nn

from core.model_run import ModelRun
from lib.utils.gpu_batcher import GPUBatcher
from models.baseline_model import BaselineModel




class ChampionModel(BaselineModel):
    def __init__(self, config: dict, network_cls: type[nn.Module]):
        super().__init__(config)
        self.network_cls = network_cls
        self.name        = network_cls.__name__.lower()
        self.type        = "champion"
        self.MAX_TORCH_SEED = 2**64 - 1
        self.SKIPPING_TEXT = Fore.YELLOW + "[SKIPPING]" + Style.RESET_ALL
        self.TRAINING_TEXT = Fore.BLUE + "[TRAINING]" + Style.RESET_ALL
        self.device      = self.get_device()
        self.seeds       = self.get_seeds()
        self.configure_torch()

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_seeds(self) -> list[int]:
        with open(f"{self.config['admin']['data_folder_path']}/seeds.txt", "r") as f:
            seeds = [int(seed) for seed in f.read().split()][:int(self.config["training"]["n_seeds"])]

        too_large = [seed for seed in seeds if seed > self.MAX_TORCH_SEED]
        if too_large:
            raise ValueError(f"seeds.txt values must be <= {self.MAX_TORCH_SEED} for torch.manual_seed: {too_large}")
        return seeds

    def configure_torch(self) -> None:
        ModelRun.configure_torch(self.config)

    def get_model_name(self, seed_index: int) -> str:
        return f"{self.name}-{seed_index:02d}"

    def seed_is_complete(self, path: str, seed_index: int) -> bool:
        results_file = Path(path) / self.get_model_name(seed_index) / "results.txt"
        return results_file.is_file() and "Parameters =" in results_file.read_text()

    def get_remaining_seed_groups(self, path: str) -> list[list[int]]:
        remaining = [i for i in range(len(self.seeds)) if not self.seed_is_complete(path, i)]
        completed = [self.get_model_name(i) for i in range(len(self.seeds)) if i not in remaining]
        if completed:
            print(f"{self.SKIPPING_TEXT} {', '.join(completed)}: results already exist in {path}")

        group_size = int(self.config["training"]["models_in_parallel"])
        return [remaining[g:g + group_size] for g in range(0, len(remaining), group_size)]

    def get_splits(self, data: pd.DataFrame, prediction_time: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        train, targets_train, test, targets_test = self.pair_input_output(
            data, self.config["data"]["use_phases"], prediction_time)
        return train.transpose(0, 2, 1), targets_train, test.transpose(0, 2, 1), targets_test

    def build_network(self, seed: int, input_size: int) -> nn.Module:
        torch.manual_seed(seed)
        accepted = inspect.signature(self.network_cls.__init__).parameters
        kwargs   = {k: v for k, v in self.config["model"].items() if k in accepted}
        return self.network_cls(input_size=input_size, **kwargs).to(self.device)

    def set_up_models_train_test(self, train: np.ndarray, targets_train: np.ndarray, test: np.ndarray,
                                 targets_test: dict, seed_indices: list[int]
                                 ) -> tuple[list[nn.Module], list[str], GPUBatcher, GPUBatcher, nn.Module]:
        input_size = train.shape[2]
        batch_size = self.config["training"]["batch_size"]

        models      = [self.build_network(self.seeds[i], input_size) for i in seed_indices]
        model_names = [self.get_model_name(i) for i in seed_indices]

        train_loader = GPUBatcher(train, targets_train, batch_size, self.device, shuffle=True, drop_last=True)
        test_loader  = GPUBatcher(test, np.array(list(targets_test.values())), batch_size, self.device, shuffle=False)

        return models, model_names, train_loader, test_loader, nn.MSELoss()

    def count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def record_parameters(self, model: nn.Module, path: str) -> int:
        n_parameters = self.count_parameters(model)
        with open(f"{path}/results.txt", "a") as outfile:
            outfile.write(f"Parameters = {n_parameters}\n")
        print(f"Parameters = {n_parameters}")
        return n_parameters

    def test_model(self, model: nn.Module, test_loader: GPUBatcher) -> np.ndarray:
        model.eval()
        compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
        with torch.no_grad():
            predictions = [compiled(batch["Current_Flux_Data"]).squeeze(-1).clone() for batch in test_loader]
        return torch.cat(predictions).cpu().numpy()

    def evaluate_model(self, model: nn.Module, model_name: str, train_time: timedelta, test_loader: GPUBatcher,
                       targets_test: dict, event_times: list, data: pd.DataFrame, path: str,
                       epochs: int, wall_time: timedelta, n_models: int) -> np.ndarray:
        y_pred      = self.test_model(model, test_loader)
        predictions = dict(zip(targets_test.keys(), y_pred))
        model_path  = Path(path) / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        print(f"======== Evaluating {model_name} ========")
        np.savetxt(f"{model_path}/predictions.txt", y_pred, fmt='%.7f')
        self.evaluate(targets_test, predictions, event_times, data, str(model_path), train_time)
        ModelRun.record_training(str(model_path), epochs, wall_time, n_models)
        self.record_parameters(model, str(model_path))   # last line: marks the seed as complete
        return y_pred

    def get_models_path(self, path: str) -> Path:
        models_path = Path(path) / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        return models_path

    def save_model(self, model: nn.Module, model_name: str, path: str) -> None:
        torch.save(model.state_dict(), self.get_models_path(path) / f"{model_name}.pt")

    def print_training(self, model_names: list[str], prediction_time: int, path: str) -> None:
        dataset = Path(path).parent.name
        phases  = "phases" if self.config["data"]["use_phases"] else "no phases"
        for model_name in model_names:
            print(f"{self.TRAINING_TEXT} {model_name} | t+{prediction_time} | {dataset} | {phases}")

    def run(self, data: pd.DataFrame, prediction_time: int, event_times: list, path: str) -> list[np.ndarray]:
        seed_groups = self.get_remaining_seed_groups(path)
        if not seed_groups:
            return []

        train, targets_train, test, targets_test = self.get_splits(data, prediction_time)
        event_times_test = self.test_events(event_times, targets_test)

        trainer = ModelRun(model=self, config=self.config)   # training engine (core/model_run.py)
        all_predictions = []
        for seed_indices in seed_groups:
            models, model_names, train_loader, test_loader, loss_fn = self.set_up_models_train_test(
                train, targets_train, test, targets_test, seed_indices)

            self.print_training(model_names, prediction_time, path)
            train_times, epochs, wall_time = trainer.train_models(models, model_names, train_loader, loss_fn)
            for model, model_name in zip(models, model_names):
                self.save_model(model, model_name, path)

            for model, model_name, train_time, n_epochs in zip(models, model_names, train_times, epochs):
                all_predictions.append(self.evaluate_model(model, model_name, train_time, test_loader,
                                                           targets_test, event_times_test, data, path,
                                                           n_epochs, wall_time, len(models)))
        return all_predictions
