import copy
from datetime import datetime

import numpy as np
import torch
from torch import nn

from transformer.m1dataset import GPUBatcher
from utils.models import build_model


def train_models(models, optimizers, model_names, train_loader, loss_fn, device,
                  n_epoch=1000, patience=20, min_delta=1e-4, already_done=None):
    """Train n models together as ONE vectorized computation via torch.func
    (stack_module_state + functional_call + vmap), instead of a Python loop
    calling each model separately. Per-model torch.cuda.Streams (tried twice)
    and a separately-compiled optimizer step_fn were both tried and reverted
    -- see git history on this function for why. This is the next lever:
    turn n small dispatched forward/backward calls into 1 wider one, per the
    "make kernels fill the GPU instead of interlacing streams" lesson from
    external research, rather than trying to overlap n small ones.

    `models` are still individually torch.compile(mode="reduce-overhead")'d
    by set_up_models_train_test (unchanged) -- here they're used only as (a)
    the source of n different random initializations to stack, and (b) the
    object that gets the winning weights written back into it at the end, so
    checkpoint saving/loading/testing in main() needs no changes at all.

    Simplification vs. a "real" dynamic implementation: a model that hits
    patience just stops updating its best_state/patience bookkeeping -- it
    is NOT dropped out of the vmap'd computation (that would need rebuilding
    the stack with fewer rows mid-training, real added complexity). With
    groups of 5-10 models, the wasted compute on an already-converged model
    for the rest of the group's training is cheap; only the recorded best
    checkpoint has to be right, not the amount of compute spent getting
    there. randomness='different' on vmap is required because the model has
    dropout -- vmap errors on random ops unless told each stacked row should
    draw independent randomness."""
    n_batches = len(train_loader)
    n = len(models)
    best_losses = [float("inf")] * n
    patience_counters = [0] * n
    best_states = [None] * n
    active = [not d for d in (already_done if already_done is not None else [False] * n)]

    if not any(active):
        return models, best_losses

    raw_models = [m._orig_mod if hasattr(m, "_orig_mod") else m for m in models]
    base_model = copy.deepcopy(raw_models[0]).to("meta")
    base_model.train()

    params, buffers = torch.func.stack_module_state(raw_models)
    params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}

    optimizer = torch.optim.Adam(list(params.values()), lr=optimizers[0].defaults["lr"],
                                  fused=device.type == "cuda")

    def compute_loss(p, b, x, y):
        pred = torch.func.functional_call(base_model, (p, b), (x,)).squeeze(-1)
        return loss_fn(pred, y)

    compute_losses = torch.func.vmap(compute_loss, in_dims=(0, 0, None, None), randomness="different")
    compute_losses = torch.compile(compute_losses, mode="reduce-overhead")

    def clip_grad_norm_stacked(max_norm=1.0):
        sq_sums = None
        for p in params.values():
            if p.grad is None:
                continue
            s = p.grad.reshape(n, -1).pow(2).sum(dim=1)
            sq_sums = s if sq_sums is None else sq_sums + s
        clip_coef = (max_norm / (sq_sums.sqrt() + 1e-6)).clamp(max=1.0)
        for p in params.values():
            if p.grad is None:
                continue
            p.grad.mul_(clip_coef.view(n, *([1] * (p.grad.dim() - 1))))

    for epoch in range(n_epoch):
        if not any(active):
            break
        total_losses = torch.zeros(n, device=device)
        epoch_start_time = datetime.now()

        for batch in train_loader:
            x = batch["Current_Flux_Data"]
            y = batch["Target_Proton_Data"]

            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                # nn.TransformerEncoderLayer's attention dispatches to a fused
                # (flash-attention) SDPA kernel by default, which assumes a
                # memory layout vmap's batching transform doesn't produce --
                # errors with "LSE is not correctly aligned" otherwise. MATH
                # is the plain, stride-agnostic attention implementation.
                losses = compute_losses(params, buffers, x, y)   # shape [n] -- 1 dispatch for all n models
                optimizer.zero_grad(set_to_none=True)
                losses.sum().backward()
            clip_grad_norm_stacked(max_norm=1.0)
            optimizer.step()
            total_losses += losses.detach()   # stays on GPU, no sync

        epoch_end_time = datetime.now()
        avg_losses = (total_losses / n_batches).tolist()   # one sync for all n models, not n syncs
        for i in range(n):
            if not active[i]:
                continue
            avg_loss = avg_losses[i]

            if best_losses[i] - avg_loss > min_delta:
                best_losses[i] = avg_loss
                patience_counters[i] = 0
                best_states[i] = {k: v[i].detach().clone() for k, v in params.items()}
            else:
                patience_counters[i] += 1

            print(f"{model_names[i]} Epoch {epoch+1}/{n_epoch} done in "
                  f"{(epoch_end_time - epoch_start_time).total_seconds():.1f}s — "
                  f"loss {avg_loss:.4f} | best {best_losses[i]:.4f} | "
                  f"patience {patience_counters[i]}/{patience}")

            if patience_counters[i] >= patience:
                print(f"{model_names[i]} Converged at epoch {epoch+1}")
                active[i] = False

    with torch.no_grad():
        for i, raw in enumerate(raw_models):
            if best_states[i] is None:
                continue
            for name, param in raw.named_parameters():
                param.data.copy_(best_states[i][name])

    return models, best_losses

def test_model(model, test_loader, device, loss_fn):
    model.eval()

    predictions = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch in test_loader:
            x = batch["Current_Flux_Data"]
            pred = model(x).squeeze(-1)
            predictions = torch.cat((predictions, pred.flatten()))

    return predictions

def set_up_models_train_test(train, targets_train, test, targets_test, cfg, device, dataset_id, seed_bases, seeds):
    """Build one independent model/optimizer pair per seed in `seeds` (a subset
    of range(cfg["n_seeds"]) -- how many run together is controlled by
    cfg["models_in_parallel"], not this function) plus ONE shared pair of
    loaders for one already-built dataset variant (from pair_input_output's
    n_datasets output). Doesn't call pair_input_output itself, so every seed
    trained on the same dataset variant -- and the linear baseline -- share
    the exact same train/test split; only the dataset variant differs across
    the outer loop."""

    models, optimizers, model_names = [], [], []
    for seed in seeds:
        torch.manual_seed(seed_bases[seed] + dataset_id)
        model = build_model(cfg, device)
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model, mode="reduce-overhead")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], fused=device.type == "cuda")
        models.append(model)
        optimizers.append(optimizer)
        model_names.append("M1-" + str(seed).zfill(2))

    train = np.array([instance.T for instance in train])
    train_loader = GPUBatcher(train, targets_train, batch_size=cfg["batch_size"], device=device,
                               shuffle=True, drop_last=True)

    test = np.array([instance.T for instance in test])
    targets_test = np.array(list(targets_test.values()))
    test_loader = GPUBatcher(test, targets_test, batch_size=cfg["batch_size"], device=device, shuffle=False)

    loss_fn = nn.MSELoss()

    return models, optimizers, model_names, train_loader, test_loader, loss_fn
