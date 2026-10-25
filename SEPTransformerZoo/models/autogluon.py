"""
AutoGluon champions: TFT (Temporal Fusion Transformer, trained from scratch) and Chronos-2 (pretrained, zero-shot).

Runs under Python 3.12 (.venv): AutoGluon doesn't support 3.14. You don't start it yourself:
`py -3.14 main.py train` trains the torch models, then starts config.yaml's admin.python312_path
with `main.py train --models tft,chronos2` for these two.

Why these two: they are the only AutoGluon models that take past covariates, i.e. can read electron,
electron_high (and the phase channels) for the context window like every other model here does.

Same data as every other model: the windows from BaselineModel.pair_input_output, the same train/test split,
the same seeds.txt, and results saved as results/<name>/<phases>/dataset_N/t+H/<name>-XX/.

AutoGluon forecasts a whole horizon instead of one value, so each window becomes one AutoGluon "item":
    context = the window's 25 steps (proton is the target, the other channels are past covariates)
    horizon = prediction_time steps; only the last step (t + prediction_time) is kept as the prediction,
              the same value every other model predicts for that window.
Training items also carry the real proton values for t+1 .. t+prediction_time (what AutoGluon learns from).
Covariates stay NaN past the context window, so no future electron values leak in.

No time limit: training stops on early stopping (training.patience) or training.n_epochs, like the torch models.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from colorama import Fore, Style

from models.baseline_model import BaselineModel

WINDOW_SIZE      = 25          # 24-step lookback + current, as in pair_input_output
FIRST_T          = 24          # pair_input_output's first window ends at row 24
PROTON_CHANNEL   = 2           # windows are [electron, electron_high, proton, (background, rising, falling)]
CHANNEL_NAMES    = ['electron', 'electron_high', 'proton', 'background', 'rising', 'falling']
VALIDATION_ITEMS = 2000        # AutoGluon's early stopping watches validation loss, so it needs a held-out set
MAX_AG_SEED      = 2**31 - 1   # AutoGluon's random_seed must fit in an int32


class AutoGluonModel(BaselineModel):
    ag_model = ''      # AutoGluon's name for the model
    trains   = True    # False = pretrained and used as-is: every seed would give the same forecast

    def __init__(self, config: dict):
        super().__init__(config)
        self.type          = "champion"
        self.SKIPPING_TEXT = Fore.YELLOW + "[SKIPPING]" + Style.RESET_ALL
        self.TRAINING_TEXT = Fore.BLUE + "[TRAINING]" + Style.RESET_ALL
        self.seeds         = self.get_seeds()

    def get_seeds(self) -> list[int]:
        with open(f"{self.config['admin']['data_folder_path']}/seeds.txt", "r") as f:
            seeds = [int(seed) for seed in f.read().split()][:int(self.config["training"]["n_seeds"])]
        return seeds if self.trains else seeds[:1]

    def get_model_name(self, seed_index: int) -> str:
        return f"{self.name}-{seed_index:02d}"

    def seed_is_complete(self, path: str, seed_index: int) -> bool:
        results_file = Path(path) / self.get_model_name(seed_index) / "results.txt"
        return results_file.is_file() and "Parameters =" in results_file.read_text()

    def hyperparameters(self) -> dict:
        raise NotImplementedError

    def to_frame(self, windows: np.ndarray, rows: np.ndarray, proton: np.ndarray, n_future: int) -> TimeSeriesDataFrame:
        # One item per window: 25 context steps (+ n_future real proton values for training items).
        # windows: (n, channels, 25) from pair_input_output, rows: the row t each window ends at.
        n, n_channels, _ = windows.shape
        n_total = WINDOW_SIZE + n_future

        target = windows[:, PROTON_CHANNEL, :].astype(np.float64)
        if n_future:
            future = proton[rows[:, None] + np.arange(1, n_future + 1)[None, :]]
            target = np.concatenate([target, future], axis=1)

        columns = {'item_id': np.repeat(rows, n_total),
                   'timestamp': np.tile(pd.date_range("2000-01-01", periods=n_total, freq="5min").values, n),
                   'target': target.ravel()}
        for c in range(n_channels):
            if c == PROTON_CHANNEL:
                continue
            values = np.full((n, n_total), np.nan)
            values[:, :WINDOW_SIZE] = windows[:, c, :]   # past covariate: known only up to t
            columns[CHANNEL_NAMES[c]] = values.ravel()

        return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(columns), id_column='item_id',
                                                   timestamp_column='timestamp')

    def count_parameters(self, predictor: TimeSeriesPredictor, fitted_name: str) -> int | float:
        # Number of weights in the fitted network; NaN if AutoGluon doesn't expose it
        try:
            model = predictor._learner.load_trainer().load_model(fitted_name)
            if getattr(model, '_model_pipeline', 0) is None:
                model.load_model_pipeline()   # Chronos-2 loads its weights lazily
            for attr in ('gts_predictor.prediction_net', '_model_pipeline.model', 'model'):
                net = model
                for part in attr.split('.'):
                    net = getattr(net, part, None)
                if net is not None and hasattr(net, 'parameters'):
                    return sum(p.numel() for p in net.parameters())
        except Exception:
            pass
        return float('nan')

    def fit_predict(self, seed: int, train_df: TimeSeriesDataFrame, tuning_df: TimeSeriesDataFrame,
                    test_df: TimeSeriesDataFrame, prediction_time: int, models_dir: str
                    ) -> tuple[np.ndarray, timedelta, int | float]:
        predictor = TimeSeriesPredictor(target='target', prediction_length=prediction_time, freq='5min',
                                        path=models_dir, verbosity=2)
        predictor.fit(train_df, tuning_data=tuning_df, hyperparameters={self.ag_model: self.hyperparameters()},
                      enable_ensemble=False, random_seed=seed)

        fitted_names = predictor.model_names()
        if not fitted_names:
            raise RuntimeError(f"AutoGluon didn't fit {self.ag_model}; see the log above")
        fitted = fitted_names[0]   # AutoGluon's own name can differ from ag_model (e.g. Chronos2)

        fit_seconds = predictor.leaderboard().set_index('model')['fit_time_marginal'].get(fitted, float('nan'))
        forecast = predictor.predict(test_df, model=fitted)['mean']
        # keep the last forecast step of each item = the value at t + prediction_time
        rows = test_df.item_ids.to_numpy()
        y_pred = forecast.groupby(level='item_id').last().reindex(rows).to_numpy()
        return y_pred, timedelta(seconds=float(fit_seconds)), self.count_parameters(predictor, fitted)

    def run(self, data: pd.DataFrame, prediction_time: int, event_times: list, path: str) -> list[np.ndarray]:
        remaining = [i for i in range(len(self.seeds)) if not self.seed_is_complete(path, i)]
        completed = [self.get_model_name(i) for i in range(len(self.seeds)) if i not in remaining]
        if completed:
            print(f"{self.SKIPPING_TEXT} {', '.join(completed)}: results already exist in {path}")
        if not remaining:
            return []

        # The same windows and split as every other model
        train, _, test, targets_test = self.pair_input_output(data, self.config["data"]["use_phases"], prediction_time)
        rows_train = FIRST_T + np.arange(len(train))
        rows_test  = FIRST_T + len(train) + np.arange(len(test))
        proton     = data['proton'].to_numpy(np.float64)

        train_df = self.to_frame(train, rows_train, proton, n_future=prediction_time)
        test_df  = self.to_frame(test, rows_test, proton, n_future=0)
        event_times_test = self.test_events(event_times, targets_test)

        all_predictions = []
        for i in remaining:
            seed       = self.seeds[i] % MAX_AG_SEED
            model_name = self.get_model_name(i)
            model_path = Path(path) / model_name
            model_path.mkdir(parents=True, exist_ok=True)

            # Validation items for early stopping, drawn from the training windows with this seed
            val = np.random.RandomState(seed).choice(len(train), size=min(VALIDATION_ITEMS, len(train)), replace=False)
            tuning_df = self.to_frame(train[val], rows_train[val], proton, n_future=prediction_time)

            # Fitted predictor, kept like the torch models' .pt files (Chronos-2 only stores a few KB here;
            # its pretrained weights stay in the shared Hugging Face cache)
            models_dir = str(Path(path) / "models" / model_name)
            phases = "phases" if self.config["data"]["use_phases"] else "no phases"
            print(f"{self.TRAINING_TEXT} {model_name} | t+{prediction_time} | {Path(path).parent.name} | {phases}")
            y_pred, train_time, n_parameters = self.fit_predict(seed, train_df, tuning_df, test_df,
                                                                prediction_time, models_dir)

            print(f"======== Evaluating {model_name} ========")
            np.savetxt(f"{model_path}/predictions.txt", y_pred, fmt='%.7f')
            predictions = dict(zip(targets_test.keys(), y_pred))
            self.evaluate(targets_test, predictions, event_times_test, data, str(model_path), train_time)
            with open(f"{model_path}/results.txt", "a") as outfile:
                outfile.write(f"Parameters = {n_parameters}\n")   # also marks this seed as complete
            print(f"Parameters = {n_parameters}")
            all_predictions.append(y_pred)
        return all_predictions


class TFT(AutoGluonModel):
    # Temporal Fusion Transformer, trained from scratch with the same settings as the torch champions
    ag_model = 'TemporalFusionTransformer'

    def __init__(self, config: dict):
        self.name = "tft"
        super().__init__(config)

    def hyperparameters(self) -> dict:
        return {'context_length':          self.config['model']['window_size'],
                'dropout_rate':            self.config['model']['dropout'],
                'max_epochs':              self.config['training']['n_epochs'],
                'early_stopping_patience': self.config['training']['patience'],
                'lr':                      self.config['training']['learning_rate'],
                'batch_size':              self.config['training']['batch_size']}


class Chronos2(AutoGluonModel):
    # Chronos-2 foundation model, zero-shot: no training on our data, so one seed is enough
    ag_model = 'Chronos2'
    trains   = False

    def __init__(self, config: dict):
        self.name = "chronos2"
        super().__init__(config)

    def hyperparameters(self) -> dict:
        # cross_learning=False: each window is forecast on its own, like every other model
        return {'context_length': self.config['model']['window_size'],
                'fine_tune':      False,
                'cross_learning': False}


MODELS = {'tft': TFT, 'chronos2': Chronos2}
