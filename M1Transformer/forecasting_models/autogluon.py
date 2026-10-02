"""AutoGluon TimeSeries zoo models, wired into the same n_datasets/n_seeds
bootstrap / evaluate() / score_forecast pipeline as every other model.type
(see main.py) -- one independently-seeded predictor.fit() per (zoo model,
prediction_time, dataset variant, seed), exactly mirroring how rnn/sin/rope/
zero each get n_seeds independent fits per dataset variant. Since there are
multiple zoo models here (not 1 architecture, see ZOO_MODELS/MODEL_NAMES),
each one gets treated as its own model_type under the hood -- see
run_autogluon_pipeline/zoo_cfg below -- so results land in one separate
results/autogluon_<name>/... tree per zoo model, each with its own seeded
M1-00..M1-{n_seeds-1} slots and its own overall_results/{results,f1}.csv,
the same shape every other model_type already produces.
utils/plot_overall_comparison.py and friends therefore need no
autogluon-specific code to pick these up -- just add "autogluon_<name>" to
their MODELS list.

Must be run under .venv-autogluon (Python 3.10-3.13; AutoGluon doesn't support
3.14, which the rest of this repo uses) -- e.g.
    .venv-autogluon\\Scripts\\python.exe main.py
with utils/config.yaml's model.type set to "autogluon". main.py/utils/models.py
only import this module lazily, inside the AUTOGLUON_TYPE branch, so running
main.py under the regular (3.14) interpreter for every other model.type is
unaffected.

THE FRAMING PROBLEM THIS FILE SOLVES
-------------------------------------
Every other model.type answers a per-instant question: "given the last 24
5-minute steps ending at t, what's proton at t+prediction_time?", repeated at
every t in the test set -- that per-t series of predictions is what
torres.m1.evaluate() and torres.time_series_classification.score_forecast
need (O2P/O2T lag, alert-window F1). AutoGluon's native mode instead forecasts
a whole `prediction_length`-step horizon from wherever known data ends for a
given item -- a different question.

The fix: treat each (t) test/train instance as its own AutoGluon "item" --
known history = proton[t-24:t+1] (plus electron/electron_high, see below),
forecast horizon = prediction_length = prediction_time -- and keep only the
LAST predicted step (t+prediction_time), exactly the value every other
model.type predicts for that same instant. A SINGLE predictor.predict() call
is batched across every item at once (vectorized construction below), so
evaluating one seed's fit costs one fit + one predict, not one call per
window. "Item" here means "one training/test SAMPLE fed into one shared
model" -- exactly the role a row plays in any (X, y) supervised dataset, NOT
"one separate model per window". Every model below trains exactly once per
(model, horizon, dataset, seed) using every item as one of many training
examples, the same way utils/training.py's train_models fits one neural net
using every window as a training example. (This is also precisely why the
local/statistical family -- ARIMA, ETS, Theta, etc. -- was never a fit for
this pipeline: those genuinely DO fit one independent model per item, by
AutoGluon's own design, and were deliberately left out of this zoo for
exactly that reason.)

THREE CHANNELS IN, ONE CHANNEL OUT
------------------------------------
Every other model.type in this repo builds its input window from THREE
channels -- electron, electron_high, proton (torres/m1.py's _build_windows)
-- and predicts a single future proton value. `electron`/`electron_high` are
only ever read for t-24..t, never for the future (using their real future
values would be leakage no other model.type gets either). In AutoGluon's
terms that's a PAST covariate (known only in the past), not a KNOWN covariate
(known in advance for the forecast horizon too, like a calendar feature) --
confirmed directly from the installed source
(TimeSeriesPredictor.fit's docstring: "Columns of train_data except target
and those listed in known_covariates_names will be interpreted as
past_covariates"). _rows_to_frame below builds electron/electron_high as
exactly that: real values for the WINDOW_SIZE context steps, NaN beyond that
(so training items' future/horizon rows never leak real future
electron/electron_high, same rule _build_windows already follows for proton).

Past-covariate support turned out to be the rare exception, not the rule:
searched every model class in the installed package for
`_supports_past_covariates = True` and found exactly TWO matches out of 30
real forecaster classes -- TemporalFusionTransformer and Chronos-2 (see
notes/autogluon_model_selection.md for the full model-by-model review).
Every other AutoGluon model -- DeepAR, PatchTST, TiDE, WaveNet,
RecursiveTabular, DirectTabular, PerStepTabular, every local/statistical
model, Chronos v1, Toto, Toto2 -- only supports KNOWN covariates (wrong
semantics here) or none at all, and is architecturally incapable of taking
electron/electron_high as input no matter how it's configured. That's why
this zoo is now just 4 models instead of the wider one tried earlier.

ZOO
---
- TemporalFusionTransformer, Chronos-2: the only two AutoGluon models that
  can genuinely take all 3 channels, matching every other model.type in this
  repo.
- Naive, Average: trivial local baselines (predict = last observed value /
  predict = historical mean) kept as sanity-check floors, the same role
  forecasting_models/persistence.py already plays for the non-AutoGluon
  models. Unlike the rest of AutoGluon's local/statistical family (ARIMA,
  ETS, Theta, ...), these don't estimate any parameters -- there's no
  per-item "fit" expensive or meaningless enough to worry about, just
  arithmetic, so the same objection that excluded the rest of that family
  doesn't apply to these two.

SCALE WARNING
-------------
len(MODEL_NAMES) x n_datasets x len(prediction_time) x n_seeds = total fit()
calls. At the config.yaml defaults (10 x 2 x 10) and this file's 4-model zoo,
that's 4 x 10 x 2 x 10 = 800. Budget autogluon.time_limit_minutes accordingly
and consider a smaller n_datasets/n_seeds for a first real run.
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from torres.m1 import evaluate, pair_input_output
from torres.time_series_classification import score_forecast
from utils.file import _result_base, create_result_dirs
from utils.output import create_result_csv, create_f1_csv

WINDOW_SIZE = 25  # 24-step lookback + current, matching every other model.type

# Which zoo models are GluonTS-based (checks torch.cuda.is_available() itself
# and switches to accelerator="gpu" with no hyperparameter needed -- confirmed
# via a real fit, "Training on device 'gpu'" in verbose logs, once
# .venv-autogluon had a CUDA build of torch installed: pip installs a CPU-only
# build by default, `pip install torch==<matching version>+cuXXX
# --index-url https://download.pytorch.org/whl/cuXXX`, matched to the CUDA
# version `nvidia-smi` reports) and therefore shares GluonTS's
# context_length/early_stopping_patience/max_epochs/dropout_rate hyperparameter
# names. Chronos-2 is NOT a GluonTS model (a separate pretrained-model family,
# see models/chronos/chronos2.py) and doesn't share these names -- it gets its
# own handling below. Naive/Average are local models with none of this at all.
GLUONTS_TYPES = {"TemporalFusionTransformer"}

# Every GLUONTS_TYPES model here happens to expose dropout_rate (confirmed
# against the installed source) -- kept as its own set anyway, separate from
# GLUONTS_TYPES, since that won't stay true if more GluonTS models are added
# back (SimpleFeedForward/DLinear/PatchTST/WaveNet don't have this parameter
# at all).
DROPOUT_TYPES = {"TemporalFusionTransformer"}

# Models whose context_length should track model.window_size (below) --
# GLUONTS_TYPES plus Chronos-2, which also has a context_length parameter
# (just a much larger native default, 8192, sized for its pretraining regime
# rather than any particular downstream task).
CONTEXT_LENGTH_TYPES = GLUONTS_TYPES | {"Chronos-2"}

ZOO_MODELS = {
    "TemporalFusionTransformer": {},
    # fine_tune=False: zero-shot for this first pass -- Chronos-2 has never
    # seen SEP-event dynamics, so this is a genuinely different kind of
    # baseline than every other model.type here (all of which train from
    # scratch on data/data.csv), not a drop-in seventh entry. Revisit once
    # zero-shot results are in; see notes/autogluon_model_selection.md for
    # the full fine_tune/cross_learning tradeoff discussion.
    # cross_learning=False: Chronos-2's default (True) makes predictions for
    # one item depend on which OTHER items land in the same batch -- every
    # other model.type in this repo (including TFT) treats each window as a
    # fully independent sample, so this keeps Chronos-2 consistent with that
    # instead of introducing a batch-order dependency nothing else has.
    "Chronos-2": {"fine_tune": False, "cross_learning": False},
    "Naive": {},
    "Average": {},
}

# Fixed order only for iteration stability in run_autogluon_pipeline -- unlike
# the earlier "whole zoo in one fit()" design, M1-XX no longer identifies
# WHICH zoo model a result is (that's now the results/autogluon_<name>/...
# directory itself); M1-XX means "seed" here, same as everywhere else.
MODEL_NAMES = sorted(ZOO_MODELS)

# Size of the explicit tuning_data handed to predictor.fit() -- see
# run_autogluon_baseline. Every OTHER model_type in this repo trains on 100%
# of train_data with no held-out split, monitoring TRAINING loss for early
# stopping (see utils/training.py's train_models) -- AutoGluon's API supports
# neither: passing num_val_windows=0 with no tuning_data is rejected outright
# ("All elements of num_val_windows must be positive integers", confirmed by
# calling predictor.fit() that way directly), and every GLUONTS_TYPES model's
# early_stopping_patience (default 20 -- see PATIENCE note on
# run_autogluon_baseline) watches VALIDATION loss via a Lightning EarlyStopping
# callback, not training loss -- there's no exposed way to switch that to
# train-loss monitoring. So tuning_data isn't just an API formality here, it's
# the actual early-stopping signal for every GLUONTS_TYPES zoo model, and needs to be
# big enough to be a meaningful signal, not just big enough to avoid an error.
# Sized well below the full ~470k-item train_data (confirmed on a real run
# that AutoGluon's own auto-derived validation windows -- what you get if you
# DON'T pass tuning_data -- predict across the entire train set and dwarf
# actual training: DeepAR took 27s to train but 245s just to validate) but
# well above a throwaway handful: 2_000 items costs a low single-digit number
# of seconds of validation-predict per model while giving early stopping a
# real sample to judge convergence from. May overlap train_data -- no leakage
# concern for OUR purposes since we don't use AutoGluon's own validation
# SCORE for model ranking (only one model is ever fit per predictor), only
# its early-stopping decisions and fit_time_marginal bookkeeping.
VALIDATION_ITEMS = 2000


def _rows_to_frame(rows, prediction_time, proton, electron, electron_high, n_future):
    """Vectorized construction of a TimeSeriesDataFrame with one item per row
    index in `rows`. Item `r`'s target series is proton[t-24 : t+1+n_future],
    where t = r - prediction_time -- i.e. the same 25-step lookback every
    other model_type uses, optionally followed by n_future more real values.
    Also builds electron/electron_high as PAST covariate columns: real values
    for the same t-24..t context window, NaN beyond that -- see the module
    docstring's THREE CHANNELS IN, ONE CHANNEL OUT section for why NaN (not
    real future values) is the correct choice there. AutoGluon infers
    "past covariate" automatically for any column that isn't `target` and
    isn't named in known_covariates_names (we never set that), so no other
    wiring is needed here for a model that actually supports them (see
    CONTEXT_LENGTH_TYPES/GLUONTS_TYPES); models that don't (Naive/Average)
    simply never read these two extra columns.

    n_future=0 (test/inference): just the 25 known-history steps; AutoGluon
    forecasts the rest itself.
    n_future=prediction_time (train): the full known TARGET span too, so
    fit() has an actual t+1..t+prediction_time sequence to learn from --
    _build_windows() elsewhere only keeps the single t+prediction_time point
    (that's the target for every other model_type), not the intermediate
    steps AutoGluon's horizon-based training needs. Covariates do NOT get
    this treatment -- see above, they stay NaN past the context window
    regardless of n_future.

    Uses a synthetic, shared 5-minute timestamp axis per item -- item_id
    keeps items independent of each other, so the real calendar time doesn't
    matter (each item is an independent window, not part of one continuous
    multi-item panel).
    """
    rows = np.asarray(rows, dtype=np.int64)
    n_total = WINDOW_SIZE + n_future
    starts = rows - prediction_time - (WINDOW_SIZE - 1)
    idx = starts[:, None] + np.arange(n_total)[None, :]

    target_values = proton[idx]
    electron_values = electron[idx].astype(np.float64)
    electron_high_values = electron_high[idx].astype(np.float64)
    if n_future > 0:
        electron_values[:, WINDOW_SIZE:] = np.nan
        electron_high_values[:, WINDOW_SIZE:] = np.nan

    offsets = pd.date_range("2000-01-01", periods=n_total, freq="5min").values
    item_id = np.repeat(rows, n_total)
    timestamp = np.tile(offsets, len(rows))

    df = pd.DataFrame({
        "item_id": item_id,
        "timestamp": timestamp,
        "target": target_values.ravel(),
        "electron": electron_values.ravel(),
        "electron_high": electron_high_values.ravel(),
    })
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")


def run_autogluon_baseline(model_name, train_rows, test_rows, targets_test, data, prediction_time,
                           result_path, event_times, models_dir=None, time_limit=None, random_seed=42,
                           patience=None, max_epochs=None, context_length=None, dropout=None):
    """Fit ONE named zoo model (one ZOO_MODELS entry) on this dataset variant
    and seed, then evaluate it into result_path -- exactly the way
    forecasting_models/persistence.py's run_persistence_baseline does for a
    single seed. Called once per (zoo model, prediction_time, dataset
    variant, seed) by run_autogluon_pipeline below.

    :param model_name: one of MODEL_NAMES -- which zoo model to fit
    :param train_rows: row indices (into `data`) of each training window's
        target -- torres.m1.pair_input_output(..., return_target_rows=True)'s
        target_rows_trains[dataset_id]
    :param test_rows: same, for the test windows -- target_rows_tests[dataset_id];
        index-aligned with targets_test's insertion order (both built from the
        same test_idx in torres.m1._partition_windows)
    :param targets_test: {timestamp_str: proton_value} for this dataset variant
    :param data: the full, untouched data/data.csv DataFrame
    :param prediction_time: forecast horizon in 5-minute steps (6 or 12)
    :param result_path: the M1-XX directory to write predictions.txt/
        results.txt/training_time.txt/model_name.txt into
    :param event_times: parsed data/event_timestamps.txt rows
    :param models_dir: where this one fitted model's weights/checkpoint are
        saved -- passed straight through as TimeSeriesPredictor(path=...).
        Defaults to AutogluonModels/ag-<timestamp>/ in the current working
        directory (AutoGluon's own default) if not given; run_autogluon_pipeline
        passes models/{base}/t+{pt}/dataset{j}/M1-XX instead, matching where
        every other model_type saves its per-seed checkpoint.
    :param time_limit: seconds for this ONE model's fit() call, or None
        (default) for NO limit -- fitting then runs until GLUONTS_TYPES
        models' own early_stopping_patience triggers (or max_epochs is hit),
        exactly like utils/training.py's train_models, which has no wall-
        clock cutoff either, only patience-based early stopping. A time_limit
        that expires before a model's own convergence criteria do just cuts
        it off mid-training with whatever it has -- worse than either
        finishing early stopping or running long, so this defaults to
        unbounded. Pass a number back in (utils/config.yaml's
        autogluon.time_limit_minutes) if you want a ceiling again.
    :param random_seed: seeds both AutoGluon's own random_seed and the draw
        of VALIDATION_ITEMS tuning rows -- pass a different value per seed
        index (see run_autogluon_pipeline) the same way seed_bases indexes
        torch.manual_seed for the other model_types.
    :param patience: overrides GLUONTS_TYPES models' early_stopping_patience
        (their own library default is 20, same NUMBER utils/training.py's
        train_models uses -- pass utils/config.yaml's training.patience here
        so that's an intentional match, not a coincidence). Ignored for
        Chronos-2/Naive/Average (no such concept -- Chronos-2 uses
        fine_tune_steps instead, only relevant if fine_tune=True). NOTE the
        mechanism still isn't identical to train_models' early stopping even
        with patience matched: GluonTS models watch VALIDATION loss (on the
        VALIDATION_ITEMS tuning set) via a Lightning EarlyStopping callback,
        not TRAINING loss like every other model_type here, and that
        callback's min_delta has no exposed hyperparameter (Lightning's own
        default, 0.0, is used -- utils/config.yaml's training.min_delta 1e-4
        has no equivalent wiring here; not fixed, since AutoGluon doesn't
        expose it and overriding it would mean reaching into internal
        trainer_kwargs construction).
    :param max_epochs: overrides GLUONTS_TYPES models' max_epochs (library
        default 100). With time_limit=None this is the only remaining
        ceiling on training length, so pass a generous value (utils/
        config.yaml's training.n_epochs, matching train_models' own n_epoch
        ceiling) rather than leaving it at 100 -- early_stopping_patience is
        meant to be what actually stops training, same as the other
        model_types, not this. Ignored for Chronos-2/Naive/Average.
    :param context_length: overrides CONTEXT_LENGTH_TYPES models'
        context_length -- pass utils/config.yaml's model.window_size here
        (same config key the custom transformer/nn/rnn model_types already
        build their own architectures from -- see utils/models.py's
        build_model), so every model_type in this repo answers from the same
        lookback instead of each model's own library default (TFT's is
        max(64, 2*prediction_length); Chronos-2's is 8192, its native
        pretraining context). Ignored for Naive/Average (no such concept).
    :param dropout: overrides DROPOUT_TYPES models' dropout_rate -- pass
        utils/config.yaml's model.dropout here, the same config key the
        custom transformer model_types use. Only TemporalFusionTransformer
        exposes a dropout_rate hyperparameter in this zoo (confirmed against
        the installed source); ignored for Chronos-2/Naive/Average.
    """
    proton = data["proton"].values
    electron = data["electron"].values
    electron_high = data["electron_high"].values

    train_data = _rows_to_frame(train_rows, prediction_time, proton, electron, electron_high,
                                n_future=prediction_time)

    rng = np.random.RandomState(random_seed)
    train_rows_arr = np.asarray(train_rows)
    val_rows = rng.choice(train_rows_arr, size=min(VALIDATION_ITEMS, len(train_rows_arr)), replace=False)
    tuning_data = _rows_to_frame(val_rows, prediction_time, proton, electron, electron_high,
                                 n_future=prediction_time)

    test_data = _rows_to_frame(test_rows, prediction_time, proton, electron, electron_high, n_future=0)

    model_hyperparameters = dict(ZOO_MODELS[model_name])
    if model_name in GLUONTS_TYPES:
        if patience is not None:
            model_hyperparameters["early_stopping_patience"] = patience
        if max_epochs is not None:
            model_hyperparameters["max_epochs"] = max_epochs
        if dropout is not None and model_name in DROPOUT_TYPES:
            model_hyperparameters["dropout_rate"] = dropout
    if context_length is not None and model_name in CONTEXT_LENGTH_TYPES:
        model_hyperparameters["context_length"] = context_length

    predictor = TimeSeriesPredictor(
        target="target", prediction_length=prediction_time, freq="5min", verbosity=2,
        path=models_dir,
    )
    predictor.fit(train_data, tuning_data=tuning_data, hyperparameters={model_name: model_hyperparameters},
                 time_limit=time_limit, enable_ensemble=False, random_seed=random_seed)

    fitted_names = predictor.model_names()
    if not fitted_names:
        raise RuntimeError(
            f"AutoGluon didn't finish fitting {model_name} (time_limit={time_limit}). "
            f"Check the log above for why -- with time_limit=None this should only happen on a "
            f"real error, not a timeout."
        )
    # The hyperparameters dict key (model_name, e.g. "Chronos-2") is an
    # ALIAS used only to look up the model class -- confirmed empirically
    # that the fitted instance's own registered name can differ from it
    # (Chronos-2 fits as "Chronos2", its class-derived name, since
    # "Chronos-2" is only a secondary ag_model_aliases entry, not the
    # primary key AutoGluon assigns internally). We fit exactly one model
    # per predictor call (single-key hyperparameters, enable_ensemble=False),
    # so predictor.model_names() has exactly one real entry -- use it
    # directly instead of assuming it matches model_name.
    fitted_name = fitted_names[0]
    fit_end = datetime.now()
    duration = predictor.leaderboard().set_index("model")["fit_time_marginal"].get(fitted_name, float("nan"))

    os.makedirs(result_path, exist_ok=True)
    # M1-XX is an anonymous seed slot everywhere else in this repo -- record
    # which zoo model it is too, for convenience (the parent directory name
    # already says so via results/autogluon_<name>/..., see zoo_cfg).
    with open(f"{result_path}/model_name.txt", "w") as f:
        f.write(model_name + "\n")

    test_rows_arr = np.asarray(test_rows, dtype=np.int64)
    preds = predictor.predict(test_data, model=fitted_name)["mean"]
    y_pred = preds.groupby(level="item_id").last().reindex(test_rows_arr).to_numpy()
    np.savetxt(f"{result_path}/predictions.txt", y_pred, delimiter=",")

    event_times_test = [event for event in event_times if event[0] in targets_test]
    target_times = list(targets_test.keys())
    predictions = {target_times[k]: y_pred[k] for k in range(len(y_pred))}
    evaluate(targets_test, predictions, event_times_test, data, path=result_path, display=False)

    fit_start = fit_end - timedelta(seconds=duration)
    with open(f"{result_path}/training_time.txt", "w") as f:
        f.write(f"Start = {fit_start.isoformat()}\n")
        f.write(f"End = {fit_end.isoformat()}\n")
        f.write(f"Duration (s) = {duration:.3f}\n")

    return predictor


def zoo_cfg(cfg, zoo_name):
    """A shallow cfg copy namespaced to one zoo model, so every existing
    utility (_result_base, create_result_dirs, create_result_csv,
    score_forecast, create_f1_csv) treats each zoo model as its own
    independent model_type -- results/autogluon_<name>/... exactly like
    results/rnn/..., results/sin/..., etc. -- so plot_overall_comparison.py
    and friends need no autogluon-specific branch to pick these up (just add
    "autogluon_<name>" to their MODELS list)."""
    model_cfg = dict(cfg)
    model_cfg["model_type"] = f"autogluon_{zoo_name.lower()}"
    model_cfg["run_tag"] = f"{model_cfg['model_type']}_{'phases' if cfg['use_phases'] else 'nophases'}"
    return model_cfg


def run_autogluon_pipeline(cfg, data, event_times, seed_bases):
    """Top-level entry point for model.type: autogluon, called once from
    main.py in place of its usual per-model_type training loop.

    Runs the FULL n_datasets x prediction_time x n_seeds bootstrap
    independently for EVERY zoo model in MODEL_NAMES -- i.e. this is
    len(MODEL_NAMES) complete sub-pipelines, one per zoo architecture, each
    behaving exactly like a normal model_type run (own results/models tree
    via zoo_cfg, own seeded M1-00..M1-{n_seeds-1} slots, own
    overall_results/{results,f1}.csv) -- matching how rnn/sin/rope/zero each
    get n_seeds independently-seeded fits per dataset variant, just fanned
    out across several different AutoGluon architectures instead of 1 custom
    one. See the module docstring's SCALE WARNING: this is len(MODEL_NAMES) x
    n_datasets x len(prediction_time) x n_seeds total fit() calls.
    """
    for zoo_name in MODEL_NAMES:
        model_cfg = zoo_cfg(cfg, zoo_name)
        base = _result_base(model_cfg)
        tag = model_cfg["run_tag"]
        create_result_dirs(model_cfg)

        f1_records = []
        for pt in cfg["prediction_time"]:
            trains, targets_trains, tests, targets_tests, target_rows_trains, target_rows_tests = pair_input_output(
                data, cfg["use_phases"], int(pt), n_datasets=cfg["n_datasets"], train_split=cfg["train_split"],
                return_target_rows=True)

            for dataset_id in range(cfg["n_datasets"]):
                print(f"======== AutoGluon/{zoo_name}: dataset {dataset_id}/{cfg['n_datasets'] - 1}, "
                      f"t+{pt} ({len(trains[dataset_id])} train / {len(tests[dataset_id])} test windows) ========")

                for seed in range(cfg["n_seeds"]):
                    model_name = "M1-" + str(seed).zfill(2)
                    result_path = f"results/{base}/resutls_per_dataset/t+{pt}/{tag}/dataset{dataset_id}/{model_name}"
                    models_dir = f"models/{base}/t+{pt}/dataset{dataset_id}/{model_name}"
                    # Matches utils/training.py's set_up_models_train_test
                    # exactly: torch.manual_seed(seed_bases[seed] + dataset_id)
                    # -- offsetting by dataset_id is what stops a given seed
                    # index from reusing the identical draw across all 10
                    # dataset variants. seed_bases entries are arbitrary-
                    # precision Python ints (kept for continuity with those
                    # torch.manual_seed calls) but AutoGluon's random_seed
                    # expects a normal-sized int, hence the modulo.
                    seed_val = (seed_bases[seed] + dataset_id) % (2 ** 31 - 1)

                    # results.txt is the last thing run_autogluon_baseline
                    # writes for a slot (via evaluate(), after predictions.txt
                    # already succeeded) -- its presence means this exact
                    # (zoo model, horizon, dataset, seed) already finished
                    # cleanly on a prior run, so skip re-fitting it. Matches
                    # main.py's own already_done check for the torch model
                    # types (os.path.exists(model_paths)), just keyed off the
                    # result file instead of a checkpoint file since AutoGluon
                    # owns its own checkpoint directory layout under models_dir.
                    if os.path.exists(f"{result_path}/results.txt"):
                        print(f"---- {zoo_name} seed {seed} ({model_name}) already done, skipping ----")
                        continue

                    print(f"---- {zoo_name} seed {seed} ({model_name}) ----")
                    run_autogluon_baseline(
                        zoo_name, target_rows_trains[dataset_id], target_rows_tests[dataset_id],
                        targets_tests[dataset_id], data, pt, result_path, event_times,
                        models_dir=models_dir, time_limit=cfg["autogluon_time_limit_seconds"],
                        random_seed=seed_val, patience=cfg["patience"], max_epochs=cfg["n_epochs"],
                        context_length=cfg["window_size"], dropout=cfg["dropout"])

                if dataset_id == 0:
                    for app in ["app0", "app1", "app2", "app3"]:
                        print(app)
                        f1_records.extend(score_forecast(pt, app, tag, base, n_seeds=cfg["n_seeds"]))

        create_result_csv(model_cfg)
        create_f1_csv(f1_records, base)
