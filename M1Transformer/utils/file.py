import os

import torch


def ensure_dir(path):
    """Create path if it doesn't already exist, so writes into it never error."""
    os.makedirs(path, exist_ok=True)

def _result_base(cfg):
    """model_type/split_type/phases_dir prefix matching the on-disk layout:
    models/{base}/t+{pt}/dataset{j}/{model_name}.pt
    results/{base}/resutls_per_dataset/t+{pt}/{run_tag}/dataset{j}/{model_name or "linear"}/...
    results/{base}/overall_results/...
    split_type is "multi-split" if n_datasets > 1, else "single-split" (kept
    separate so old single-split runs never mix with the new dataset-variant
    ones). phases_dir is "phases"/"no-phases" -- hyphenated, distinct from
    run_tag's "nophases"."""
    split_type = "multi-split" if cfg["n_datasets"] > 1 else "single-split"
    phases_dir = "phases" if cfg["use_phases"] else "no-phases"
    return f"{cfg['model_type']}/{split_type}/{phases_dir}"

def create_result_dirs(cfg):
    """Create models/{base}/t+{pt}/dataset{j},
    results/{base}/resutls_per_dataset/t+{pt}/{run_tag}/dataset{j}/{model_name}
    (one per seed) for every prediction_time and dataset variant in cfg, so
    evaluate() has somewhere to write before it's called. For model_type
    "linear" n_seeds is pinned to 1, so this makes exactly one M1-00 dir per
    dataset variant."""
    tag = cfg["run_tag"]
    base = _result_base(cfg)
    for pt in cfg["prediction_time"]:
        for dataset_id in range(cfg["n_datasets"]):
            os.makedirs(f"models/{base}/t+{pt}/dataset{dataset_id}", exist_ok=True)
            for seed in range(cfg["n_seeds"]):
                model_name = "M1-" + str(seed).zfill(2)
                os.makedirs(f"results/{base}/resutls_per_dataset/t+{pt}/{tag}/dataset{dataset_id}/{model_name}", exist_ok=True)

def load_checkpoint_safely(model, path):
    """Load a state dict that may or may not have a torch.compile '_orig_mod.' prefix,
    into a model that may or may not currently be compiled."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint at {path}. Set training.train_new_models: true in "
            f"utils/config.yaml to train one for this model.type/use_phases combination, "
            f"or double check model.type/use_phases match a run that's already been trained."
        )
    raw_sd = torch.load(path)

    # detect whether the FILE has the prefix
    file_has_prefix = any(k.startswith("_orig_mod.") for k in raw_sd.keys())

    # detect whether the MODEL currently expects the prefix (i.e., is compiled)
    model_is_compiled = hasattr(model, "_orig_mod")
    target = model._orig_mod if model_is_compiled else model

    if file_has_prefix:
        # strip it so it matches the plain (uncompiled) module's key names
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}

    target.load_state_dict(raw_sd)
    return model
