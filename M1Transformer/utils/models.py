import yaml

from transformer.m1transformersin import M1TransformerSin
from transformer.m1transfotrmerRoPE import M1TransformerRoPE
from transformer.m1transformerzero import M1TransformerZero
# from transformer.m1transformerT5 import M1TransformerT5

MODEL_REGISTRY = {
    "sin": M1TransformerSin,
    "rope": M1TransformerRoPE,
    "zero": M1TransformerZero,
    # "t5": M1TransformerT5,
}

# model.type == "linear" selects the sklearn LinearRegression baseline instead
# of a transformer architecture. It's a peer model type now (same config knob,
# same on-disk layout via _result_base), not a separate training_linear toggle.
# It has no seeds/epochs/checkpoints, so load_config forces n_seeds and
# models_in_parallel to 1 for it -- the aggregation code in utils/output.py
# then treats it as a one-seed run (single M1-00 directory) with no changes.
LINEAR_TYPE = "linear"
VALID_MODEL_TYPES = set(MODEL_REGISTRY) | {LINEAR_TYPE}

def load_config(path="utils/config.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)

    model_type = config["model"]["type"]
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Unknown model.type '{model_type}', expected one of {sorted(VALID_MODEL_TYPES)}")
    use_phases = config["data"]["use_phases"]

    is_linear = model_type == LINEAR_TYPE
    # Linear has no seeds -- pin these to 1 regardless of what's in the yaml so
    # every "loop over n_seeds" downstream sees exactly one M1-00 run.
    n_seeds = 1 if is_linear else config["training"]["n_seeds"]
    models_in_parallel = 1 if is_linear else config["training"]["models_in_parallel"]

    cfg = {

        "model_type":       model_type,
        "n_encoder_layers": config["model"]["n_encoder_layers"],
        "dim_val":       config["model"]["dim_val"],
        "dim_feedforward":  config["model"]["dim_feedforward"],
        "n_heads":          config["model"]["n_heads"],
        "dropout":          config["model"]["dropout"],
        "window_size":           config["model"]["window_size"],

        "learning_rate":   config["training"]["learning_rate"],
        "n_epochs":         config["training"]["n_epochs"],
        "patience":         config["training"]["patience"],
        "min_delta":        config["training"]["min_delta"],
        "batch_size":      config["training"]["batch_size"],
        "train_new_models": config["training"]["train_new_models"],
        "plot_only":        config["training"]["plot_only"],
        "n_seeds":          n_seeds,
        "models_in_parallel": models_in_parallel,
        "n_datasets":       config["training"]["n_datasets"],

        "prediction_time":    config["data"]["prediction_time"],
        "train_split":      config["data"]["train_split"],
        "use_phases":       use_phases,

        # Namespaces models/ and results/ so switching model type or phases
        # never silently overwrites a previous run's checkpoints/predictions.
        "run_tag": f"{model_type}_{'phases' if use_phases else 'nophases'}",
    }
    return cfg


def build_model(cfg, device):
    """Construct the configured architecture. RoPE/T5 take d_model + dim_feedforward
    explicitly; Sin/Zero take dim_val and derive dim_val*4 as the feedforward size
    internally (which equals cfg["dim_feedforward"]'s default of 64)."""
    model_cls = MODEL_REGISTRY[cfg["model_type"]]
    input_size = 6 if cfg["use_phases"] else 3

    kwargs = dict(
        input_size=input_size,
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_encoder_layers"],
        dropout=cfg["dropout"],
        window_size=cfg["window_size"],
    )
    if cfg["model_type"] in ("rope", "t5"):
        kwargs["d_model"] = cfg["dim_val"]
        kwargs["dim_feedforward"] = cfg["dim_feedforward"]
    else:
        kwargs["dim_val"] = cfg["dim_val"]

    return model_cls(**kwargs).to(device)
