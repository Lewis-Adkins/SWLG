import copy
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd
import torch
from torch import nn

from lib.utils.gpu_batcher import GPUBatcher
from lib.utils.paths import results_path as build_results_path


class ModelRun:
    # One model on one dataset/horizon: loads the data, builds the results path, and (for champion models) is the
    # training engine. The seeds of a group are stacked into one vmapped model and trained together; per epoch
    # either the normal step (compiled loss/backward, eager clip + Adam) or, with training.full_cuda_graph, one
    # CUDA-graph replay of the whole step. ChampionModel builds the networks, data loaders and evaluation and
    # calls train_models() here.
    def __init__(self, model: object, config: dict):
        self.model  = model
        self.config = config
        self.device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def get_data(self, data_index: int) -> pd.DataFrame:
        self.data = pd.read_csv(f"data/data_{data_index}.csv")
        return self.data

    def create_results_path(self, dataset_id: int, prediction_time: int) -> str:
        phases_tag   = "phases" if self.config["data"]["use_phases"] else "nophases"
        # layout defined in lib/utils/paths.py (Analysis parses it back with parse_results_path)
        results_path = build_results_path(self.config["admin"]["results_folder_path"], self.model.name, phases_tag,
                                          dataset_id, prediction_time)
        results_path.mkdir(parents=True, exist_ok=True)
        return str(results_path)

    # ---------- torch setup ----------
    @staticmethod
    def configure_torch(config: dict) -> None:
        torch.set_float32_matmul_precision("high")
        n_runs = (int(config["training"]["n_seeds"]) * int(config["training"]["n_datasets"])
                  * len(config["data"]["prediction_time"]))
        torch._dynamo.config.recompile_limit = n_runs + 16

    def use_full_cuda_graph(self) -> bool:
        # training.full_cuda_graph: capture the whole step (loss, backward, clip, Adam) in one CUDA graph
        return self.device.type == "cuda" and bool(self.config["training"].get("full_cuda_graph", False))

    # ---------- stacked (vmapped) seeds ----------
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
        # With the full CUDA graph the default compile mode is used: reduce-overhead captures CUDA graphs of its
        # own, which can't be nested inside the manual whole-step graph.
        mode = "default" if self.use_full_cuda_graph() else "reduce-overhead"
        return torch.compile(compute_losses, mode=mode, dynamic=False)

    def clip_grad_norm_stacked(self, params: dict, n_models: int, max_norm: float = 1.0) -> None:
        grads   = [p.grad for p in params.values() if p.grad is not None]
        sq_sums = sum(g.reshape(n_models, -1).pow(2).sum(dim=1) for g in grads)
        clip    = (max_norm / (sq_sums.sqrt() + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(clip.view(n_models, *([1] * (g.dim() - 1))))

    # ---------- one epoch ----------
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

    def capture_train_step(self, compute_losses: Callable, params: dict, buffers: dict,
                           optimizer: torch.optim.Optimizer, train_loader: GPUBatcher, n_models: int) -> dict:
        # Records one whole training step (loss + backward + gradient clipping + Adam) as a CUDA graph, so each
        # batch costs one graph.replay() instead of dozens of small kernel launches (~5x faster for sin at
        # batch 32). The graph reads its inputs from fixed buffers, so each batch is copied into them first
        # (drop_last=True keeps every batch the same shape). Needs Adam(capturable=True).
        # Input slots only need the right shape (each real batch is copied in before every replay), so take the
        # first batch_size rows directly: next(iter(train_loader)) would run a shuffle and use up random numbers.
        bs = train_loader.batch_size
        static = {"x": train_loader.data[:bs].clone(), "y": train_loader.targets[:bs].clone()}

        def step() -> torch.Tensor:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                losses = compute_losses(params, buffers, static["x"], static["y"])
                losses.sum().backward()
            self.clip_grad_norm_stacked(params, n_models)
            optimizer.step()
            return losses

        # Warm-up steps (compile + create autograd/optimizer state) must run before capture. They update the
        # weights, so save them first and undo the warm-up afterwards: training then starts from the same
        # initial weights and a fresh Adam state, exactly as without the graph.
        # The warm-up also draws dropout masks, so save the random-number state too and restore it at the end:
        # training then starts with the same shuffle/dropout sequence as without the graph.
        rng_state = torch.cuda.get_rng_state(self.device)
        initial = {k: v.detach().clone() for k, v in params.items()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                step()
        torch.cuda.current_stream().wait_stream(side)

        # Capture on the same side stream as the warm-up: by default torch.cuda.graph() uses its own stream, and the
        # parameters' AccumulateGrad nodes made during warm-up would then be on a different stream from the
        # captured backward ("AccumulateGrad node's stream does not match" warning). detach() drops the captured
        # step's autograd graph, so static["losses"] doesn't keep those nodes alive either.
        graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph, stream=side):   # recorded, not executed
            static["losses"] = step().detach()

        with torch.no_grad():                    # undo the warm-up (in place: the graph keeps these tensors)
            for k, v in params.items():
                v.copy_(initial[k])
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()           # exp_avg, exp_avg_sq and the step count
        torch.cuda.set_rng_state(rng_state, self.device)   # undo the warm-up's random draws
        static["graph"] = graph
        return static

    def train_epoch_graph(self, static: dict, train_loader: GPUBatcher, n_models: int) -> torch.Tensor:
        total_losses = torch.zeros(n_models, device=self.device)
        for batch in train_loader:
            static["x"].copy_(batch["Current_Flux_Data"])
            static["y"].copy_(batch["Target_Proton_Data"])
            static["graph"].replay()
            total_losses += static["losses"].detach()
        return total_losses / len(train_loader)

    # ---------- early stopping and per-model timing ----------
    def new_tracker(self, n_models: int) -> dict:
        return {"best_losses": [float("inf")] * n_models, "patience": [0] * n_models,
                "best_states": [None] * n_models, "active": [True] * n_models,
                "train_times": [None] * n_models,
                "compute_seconds": [0.0] * n_models, "epochs": [0] * n_models}

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

    def charge_epoch(self, tracker: dict, epoch_time: float) -> None:
        # All seeds train together in one vmapped step, so there is no separate clock per model. Each model is
        # charged its share of the epoch: epoch time / number of models still training (converged models keep
        # riding along in the vmap but are no longer charged). Summed over epochs this is the model's compute time.
        active = [i for i, a in enumerate(tracker["active"]) if a]
        for i in active:
            tracker["compute_seconds"][i] += epoch_time / len(active)
            tracker["epochs"][i] += 1

    def load_best_states(self, models: list[nn.Module], best_states: list[dict | None]) -> None:
        with torch.no_grad():
            for model, state in zip(models, best_states):
                if state is None:
                    continue
                for name, param in model.named_parameters():
                    param.copy_(state[name])

    @staticmethod
    def record_training(path: str, epochs: int, wall_time: timedelta, n_models: int) -> None:
        # Epochs this model trained for, and the wall-clock time of the whole group it was trained with
        # (names avoid the substring "Train Time" so the analysis parser still reads the per-model Train Time)
        with open(f"{path}/results.txt", "a") as outfile:
            outfile.write(f"Epochs = {epochs}\n")
            outfile.write(f"Group Wall Clock = {wall_time} ({n_models} models in parallel)\n")

    # ---------- training loop ----------
    def train_models(self, models: list[nn.Module], model_names: list[str], train_loader: GPUBatcher,
                     loss_fn: nn.Module) -> tuple[list[timedelta], list[int], timedelta]:
        # Trains a group of seeds together. Returns each model's Train Time (its share of the compute), the
        # epochs each trained, and the group's wall-clock time; the models end up holding their best states.
        n_models = len(models)
        base_model, params, buffers = self.stack_models(models)
        compute_losses = self.make_compute_losses(base_model, loss_fn)
        full_graph = self.use_full_cuda_graph()
        optimizer = torch.optim.Adam(list(params.values()), lr=self.config["training"]["learning_rate"],
                                     fused=self.device.type == "cuda", capturable=full_graph)
        tracker = self.new_tracker(n_models)

        start_time = datetime.now()
        static = (self.capture_train_step(compute_losses, params, buffers, optimizer, train_loader, n_models)
                  if full_graph else None)
        for epoch in range(self.config["training"]["n_epochs"]):
            if not any(tracker["active"]):
                break
            epoch_start = datetime.now()
            if full_graph:
                avg_losses = self.train_epoch_graph(static, train_loader, n_models).tolist()
            else:
                avg_losses = self.train_epoch(compute_losses, params, buffers, optimizer, train_loader,
                                              n_models).tolist()
            epoch_time = (datetime.now() - epoch_start).total_seconds()   # .tolist() above synced the GPU
            self.charge_epoch(tracker, epoch_time)
            self.update_tracker(tracker, avg_losses, params, model_names, epoch, start_time)

            if (epoch + 1) % 10 == 0:
                for model_name, loss, patience, active in zip(model_names, avg_losses, tracker["patience"],
                                                              tracker["active"]):
                    if active:
                        print(f"\t{model_name} | epoch {epoch + 1}/{self.config['training']['n_epochs']} | "
                              f"loss {loss:.4f} | epoch time {epoch_time:.1f}s | "
                              f"patience {patience}/{self.config['training']['patience']}")

        # Train Time per model = its share of the compute (see charge_epoch); the shared wall-clock time of the
        # whole group is recorded separately
        wall_time = datetime.now() - start_time
        train_times = [timedelta(seconds=s) for s in tracker["compute_seconds"]]

        self.load_best_states(models, tracker["best_states"])
        return train_times, tracker["epochs"], wall_time
