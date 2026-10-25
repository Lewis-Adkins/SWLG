import copy
import inspect
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from colorama import Fore, Style
from torch import nn

from lib.utils.gpu_batcher import GPUBatcher
from models.baseline_model import BaselineModel

MAX_TORCH_SEED = 2**64 - 1


class ChampionModel(BaselineModel):
    def __init__(self, config: dict, network_cls: type[nn.Module]):
        super().__init__(config)
        self.network_cls = network_cls
        self.name        = network_cls.__name__.lower()
        self.type        = "champion"
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

        too_large = [seed for seed in seeds if seed > MAX_TORCH_SEED]
        if too_large:
            raise ValueError(f"seeds.txt values must be <= {MAX_TORCH_SEED} for torch.manual_seed: {too_large}")
        return seeds

    def configure_torch(self) -> None:
        torch.set_float32_matmul_precision("high")
        n_runs = (int(self.config["training"]["n_seeds"]) * int(self.config["training"]["n_datasets"])
                  * len(self.config["data"]["prediction_time"]))
        torch._dynamo.config.recompile_limit = n_runs + 16

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

    def stack_models(self, models: list[nn.Module]) -> tuple[nn.Module, dict, dict]:
        base_model = copy.deepcopy(models[0]).to("meta")
        base_model.train()

        params, buffers = torch.func.stack_module_state(models)
        params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
        return base_model, params, buffers

    def make_compute_losses(self, base_model: nn.Module, loss_fn: nn.Module) -> Callable:
        def compute_loss(p: dict, b: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            pred = torch.func.functional_call(base_model, (p, b), (x,)).squeeze(-1)
            return loss_fn(pred, y)

        compute_losses = torch.func.vmap(compute_loss, in_dims=(0, 0, None, None), randomness="different")
        # dynamic=False: compute_loss is the same function for every model in a run, so after nn/rnn
        # change shape (phases adds input features) torch.compile would switch to symbolic shapes,
        # which crashes the transformers' vmapped Linear layers. Static shapes just recompile instead.
        return torch.compile(compute_losses, mode="reduce-overhead", dynamic=False)

    def clip_grad_norm_stacked(self, params: dict, n_models: int, max_norm: float = 1.0) -> None:
        grads   = [p.grad for p in params.values() if p.grad is not None]
        sq_sums = sum(g.reshape(n_models, -1).pow(2).sum(dim=1) for g in grads)
        clip    = (max_norm / (sq_sums.sqrt() + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(clip.view(n_models, *([1] * (g.dim() - 1))))

    def train_epoch(self, compute_losses: Callable, params: dict, buffers: dict, optimizer: torch.optim.Optimizer,
                    train_loader: GPUBatcher, n_models: int) -> torch.Tensor:
        total_losses = torch.zeros(n_models, device=self.device)
        for batch in train_loader:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                losses = compute_losses(params, buffers, batch["Current_Flux_Data"], batch["Target_Proton_Data"])
                optimizer.zero_grad(set_to_none=True)
                losses.sum().backward()
            self.clip_grad_norm_stacked(params, n_models)
            optimizer.step()
            total_losses += losses.detach()
        return total_losses / len(train_loader)

    def new_tracker(self, n_models: int) -> dict:
        return {"best_losses": [float("inf")] * n_models, "patience": [0] * n_models,
                "best_states": [None] * n_models, "active": [True] * n_models,
                "train_times": [None] * n_models}

    def update_tracker(self, tracker: dict, avg_losses: list[float], params: dict, model_names: list[str],
                       epoch: int, start_time: datetime) -> None:
        for i, loss in enumerate(avg_losses):
            if not tracker["active"][i]:
                continue
            if tracker["best_losses"][i] - loss > self.config["training"]["min_delta"]:
                tracker["best_losses"][i], tracker["patience"][i] = loss, 0
                tracker["best_states"][i] = {k: v[i].detach().clone() for k, v in params.items()}
            else:
                tracker["patience"][i] += 1

            if tracker["patience"][i] >= self.config["training"]["patience"]:
                tracker["train_times"][i] = datetime.now() - start_time
                tracker["active"][i] = False
                print(f"{model_names[i]} converged at epoch {epoch + 1} after {tracker['train_times'][i]}")

    def load_best_states(self, models: list[nn.Module], best_states: list[dict | None]) -> None:
        with torch.no_grad():
            for model, state in zip(models, best_states):
                if state is None:
                    continue
                for name, param in model.named_parameters():
                    param.copy_(state[name])

    def train_models(self, models: list[nn.Module], model_names: list[str], train_loader: GPUBatcher,
                     loss_fn: nn.Module) -> list[timedelta]:
        n_models = len(models)
        base_model, params, buffers = self.stack_models(models)
        compute_losses = self.make_compute_losses(base_model, loss_fn)
        optimizer = torch.optim.Adam(list(params.values()), lr=self.config["training"]["learning_rate"],
                                     fused=self.device.type == "cuda")
        tracker = self.new_tracker(n_models)

        start_time = datetime.now()
        for epoch in range(self.config["training"]["n_epochs"]):
            if not any(tracker["active"]):
                break
            epoch_start = datetime.now()
            avg_losses  = self.train_epoch(compute_losses, params, buffers, optimizer, train_loader, n_models).tolist()
            epoch_time  = (datetime.now() - epoch_start).total_seconds()
            self.update_tracker(tracker, avg_losses, params, model_names, epoch, start_time)

            if (epoch + 1) % 10 == 0:
                for model_name, loss, patience, active in zip(model_names, avg_losses, tracker["patience"], tracker["active"]):
                    if active:
                        print(f"\t{model_name} | epoch {epoch + 1}/{self.config['training']['n_epochs']} | "
                              f"loss {loss:.4f} | epoch time {epoch_time:.1f}s | "
                              f"patience {patience}/{self.config['training']['patience']}")

        # models still active hit n_epochs without converging: they trained for the full run
        total_time = datetime.now() - start_time
        train_times = [t if t is not None else total_time for t in tracker["train_times"]]

        self.load_best_states(models, tracker["best_states"])
        return train_times

    def test_model(self, model: nn.Module, test_loader: GPUBatcher) -> np.ndarray:
        model.eval()
        compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
        with torch.no_grad():
            predictions = [compiled(batch["Current_Flux_Data"]).squeeze(-1).clone() for batch in test_loader]
        return torch.cat(predictions).cpu().numpy()

    def evaluate_model(self, model: nn.Module, model_name: str, train_time: timedelta, test_loader: GPUBatcher,
                       targets_test: dict, event_times: list, data: pd.DataFrame, path: str) -> np.ndarray:
        y_pred      = self.test_model(model, test_loader)
        predictions = dict(zip(targets_test.keys(), y_pred))
        model_path  = Path(path) / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        print(f"======== Evaluating {model_name} ========")
        np.savetxt(f"{model_path}/predictions.txt", y_pred, fmt='%.7f')
        self.evaluate(targets_test, predictions, event_times, data, str(model_path), train_time)
        self.record_parameters(model, str(model_path))
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

        all_predictions = []
        for seed_indices in seed_groups:
            models, model_names, train_loader, test_loader, loss_fn = self.set_up_models_train_test(
                train, targets_train, test, targets_test, seed_indices)

            self.print_training(model_names, prediction_time, path)
            train_times = self.train_models(models, model_names, train_loader, loss_fn)
            for model, model_name in zip(models, model_names):
                self.save_model(model, model_name, path)

            for model, model_name, train_time in zip(models, model_names, train_times):
                all_predictions.append(self.evaluate_model(model, model_name, train_time, test_loader,
                                                           targets_test, event_times_test, data, path))
        return all_predictions
