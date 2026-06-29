"""
SEP Proton Flux Forecasting with Transformers
==============================================
Replicates and extends Torres 2025 using an encoder-only transformer
(SEPTransformerM1) to forecast >10 MeV SEP proton flux from SOHO/EPHIN
electron precursor channels.

Run from the project root so that `utils/` and `data/` resolve correctly:
    python seu_predict_torres_transformer.py > run.log 2>&1
"""

import os
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sklearn as sk
from sklearn.linear_model import LinearRegression

import torch
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.flux_integration import Integrating_Flux
from utils.phase_detection import Find_Proton_Flux_Phases
from utils.merge_proton_electron import Merge_Proton_Electron
from utils.model import Evaluate_Models
from utils.get_rising_metrics import Get_Rising_Metrics
from utils.slice_rising import Slice_Rising


# =====================================================================
# CONFIG
# =====================================================================
def load_config(path="utils/config.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)

    cfg = {
        "find_best_model":  config["file"]["find_best_model"],
        "model_name_use":   config["file"]["model_name_use"],
        "n_encoder_layers": config["model"]["n_encoder_layers"],
        "dim_values":       config["model"]["dim_values"],
        "n_heads":          config["model"]["n_heads"],
        "dropout":          config["model"]["dropout"],
        "seed_experiment":  config["model"]["seed_experiment"],
        "train_new_models": config["model"]["train_new_model"],
        "learning_rates":   config["training"]["learning_rates"],
        "n_epochs":         config["training"]["n_epochs"],
        "batch_sizes":      config["training"]["batch_sizes"],
        "electron_range":   config["training"]["electron_range"],
        "window_size":      config["training"]["electron_range"],
        "num_workers":      config["training"]["num_workers"],
        "forecasts_out":    config["data"]["forecasts_out"],
        "train_split":      config["data"]["train_split"],
    }
    return cfg


# =====================================================================
# DATASET
# =====================================================================
class ParticleFluxDatasetM1(Dataset):
    def __init__(self, a_particle_flux_df, a_electron_range, a_forecast_out):
        self.e150 = a_particle_flux_df["E150"].values
        self.e300 = a_particle_flux_df["E300"].values
        self.p_tot = a_particle_flux_df["P_tot"].values
        self.electron_range = a_electron_range
        self.proton_instance_after = a_forecast_out

    def __len__(self):
        return self.p_tot.shape[0] - self.electron_range - self.proton_instance_after

    def __getitem__(self, idx):
        e150 = self.e150[idx: idx + self.electron_range]
        e300 = self.e300[idx: idx + self.electron_range]
        p_y = self.p_tot[idx: idx + self.electron_range]

        x = tc.tensor(np.column_stack([e150, e300, p_y]), dtype=tc.float32)
        y = tc.tensor(
            self.p_tot[idx + self.electron_range + self.proton_instance_after],
            dtype=tc.float32,
        )
        return {"Current_Flux_Data": x, "Target_Proton_Data": y}


# =====================================================================
# MODEL
# =====================================================================
class SEPTransformerM1(nn.Module):
    def __init__(self, input_size=3, dim_val=32, n_heads=4,
                 n_encoder_layers=2, dropout=0.1, window_size=24):
        super().__init__()
        self.input_layer = nn.Linear(input_size, dim_val)
        self.pos_encoding = nn.Parameter(torch.zeros(1, window_size, dim_val))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_val * 4,
            dropout=dropout,
            batch_first=True,
            norm_first= True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.output_layer = nn.Linear(dim_val, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.pos_encoding
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


# =====================================================================
# DATA LOADING / PREP
# =====================================================================
def merge_raw_data():
    """Merge proton + electron data into per-year CSVs (run once)."""
    proton_path = "data/enviromental/ephin-soho/5min_proton_1997-2025/5min/"
    Merge_Proton_Electron(
        proton_path=proton_path,
        e150_path="data/enviromental/ephin-soho/electron/E150_1997_2025.csv",
        e300_path="data/enviromental/ephin-soho/electron/E300_1997_2017.csv",
        output_path="data/enviromental/ephin-soho/merged",
    )


def load_merged_years(start_year, end_year):
    df = pd.DataFrame([])
    for i in range(start_year, end_year):
        year_df = pd.read_csv(
            f"data/enviromental/ephin-soho/merged/merged_{i}.csv"
        )
        df = pd.concat([df, year_df], ignore_index=True)



    return df


def prepare_data(cfg):
    """
    Load the Torres 1997-2002 window, split into train/test by EVENT count
    (21 train / 18 test, matching Torres) rather than by row-count percentage,
    log-transform, and standardize with training statistics.

    Torres splits "the first 80% of usable data" and reports 21 events in
    train and 18 in test. Because our data density differs from theirs, a
    row-count 80/20 split lands the boundary in 2002 with no strong events in
    test. Splitting at the date that reproduces the 21/18 event partition
    matches Torres's event distribution exactly, putting the strong 2001
    events in the test set as they are for Torres.
    """
    # --- Torres training/test window 1997-2002 ---
    torres_df = load_merged_years(1997, 2003)
    torres_df["Timestamp"] = pd.to_datetime(torres_df["Timestamp"])

    # --- Torres event catalog (authoritative onset/peak/end) ---
    torres_merged_phases = pd.read_csv("data/phases/torres_events_converted.csv")
    for col in ["Onset Timestamp", "Threshold Timestamp", "Peak Timestamp", "End Timestamp"]:
        if col in torres_merged_phases.columns:
            torres_merged_phases[col] = pd.to_datetime(torres_merged_phases[col])

    # --- determine split DATE that yields 21 train / 18 test events ---
    events_sorted = torres_merged_phases.sort_values("Onset Timestamp").reset_index(drop=True)
    n_train_events = cfg.get("n_train_events", 21)   # Torres: 21 train, 18 test
    split_date = events_sorted.iloc[n_train_events]["Onset Timestamp"]
    print(f"prepare_data: split date = {split_date} "
          f"({n_train_events} train events / {len(events_sorted) - n_train_events} test events)")

    # --- log-transform flux columns (clip to avoid log(0)) ---
    for col in ["E150", "E300", "P_tot"]:
        torres_df[col] = np.log(np.clip(torres_df[col], 1e-6, None))

    # --- split by date (events partition), not by row percentage ---
    train_df = torres_df[torres_df["Timestamp"] < split_date].reset_index(drop=True)
    test_df  = torres_df[torres_df["Timestamp"] >= split_date].reset_index(drop=True)
    print(f"prepare_data: train rows {len(train_df)}, test rows {len(test_df)}")

    # --- split the event catalog the same way, so each split evaluates its own events ---
    train_phases = events_sorted[events_sorted["Onset Timestamp"] < split_date].reset_index(drop=True)
    test_phases  = events_sorted[events_sorted["Onset Timestamp"] >= split_date].reset_index(drop=True)
    print(f"prepare_data: train phases {len(train_phases)}, test phases {len(test_phases)}")

    # --- standardize ALL splits with TRAINING statistics ---
    e150_mean, e150_std = train_df["E150"].mean(), train_df["E150"].std()
    e300_mean, e300_std = train_df["E300"].mean(), train_df["E300"].std()
    p_tot_mean, p_tot_std = train_df["P_tot"].mean(), train_df["P_tot"].std()

    for df in [train_df, test_df]:
        df["E150"] = (df["E150"] - e150_mean) / e150_std
        df["E300"] = (df["E300"] - e300_mean) / e300_std
        df["P_tot"] = (df["P_tot"] - p_tot_mean) / p_tot_std

    stats = {"p_tot_mean": p_tot_mean, "p_tot_std": p_tot_std}

    return {
        "train_df": train_df,
        "test_df": test_df,
        "torres_merged_phases": torres_merged_phases,  # full catalog
        "train_phases": train_phases,                  # events in train window
        "test_phases": test_phases,                    # events in test window
        "split_date": split_date,
        "stats": stats,
    }

# =====================================================================
# TRAINING (Torres-style convergence)
# =====================================================================
def train_model(model, train_loader, optimizer, loss_fn, device,
                n_epoch=1000, patience=20, min_delta=1e-4, tag=""):
    """Train one model with Torres-style convergence on training loss."""
    n_batches = len(train_loader)
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(n_epoch):
        model.train()
        total_loss = 0
        epoch_start_time = datetime.now()

        for batch_counter, batch in enumerate(train_loader, start=1):
            x = batch["Current_Flux_Data"].to(device)
            y = batch["Target_Proton_Data"].to(device)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            batch_per = 100 * batch_counter / n_batches
            print(f"\r{tag} Epoch {epoch+1}/{n_epoch} | "
                  f"batch {batch_counter}/{n_batches} ({batch_per:.1f}%) | "
                  f"running loss {total_loss/batch_counter:.4f}",
                  end="", flush=True)

        avg_loss = total_loss / n_batches
        epoch_end_time = datetime.now()

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        print(f"\r{tag} Epoch {epoch+1}/{n_epoch} done in "
              f"{(epoch_end_time - epoch_start_time).total_seconds():.1f}s — "
              f"loss {avg_loss:.4f} | best {best_loss:.4f} | "
              f"patience {patience_counter}/{patience}" + " " * 15)

        if patience_counter >= patience:
            print(f"{tag} Converged at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss


# =====================================================================
# HYPERPARAMETER SEARCH
# =====================================================================
def run_search(cfg, data, device, loss_fn):
    """Train each config (single-valued grid => one model per horizon)."""
    train_df = data["train_df"]
    electron_range = cfg["electron_range"]
    window_size = cfg["window_size"]
    num_workers = cfg["num_workers"]

    # clear old models/results
    for folder in ["models/", "results/"]:
        for file in os.listdir(folder):
            fp = os.path.join(folder, file)
            if folder == "results/" and not fp.endswith(".csv"):
                continue
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)

    dim_values = cfg["dim_values"]
    batch_sizes = cfg["batch_sizes"]
    n_epochs = cfg["n_epochs"]
    learning_rates = cfg["learning_rates"]
    forecasts_out = cfg["forecasts_out"]

    n_models = (len(dim_values) * len(batch_sizes) * len(n_epochs)
                * len(learning_rates) * len(forecasts_out))
    model_counter = 1
    model_params_df = pd.DataFrame([])

    for dv, bs, ne, lr, fo in product(dim_values, batch_sizes, n_epochs,
                                      learning_rates, forecasts_out):
        print(f"\nCreating model {model_counter} of {n_models}")
        model_name = "M1-" + str(model_counter).zfill(4)
        model_params = {
            "model_name": [model_name], "dim_value": [dv],
            "batch_size": [bs], "n_epoch": [ne],
            "learning_rate": [lr], "forecast_out": [fo],
        }

        train_dataset = ParticleFluxDatasetM1(train_df, electron_range, fo)
        train_loader = DataLoader(train_dataset, batch_size=bs,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=True)

        model = SEPTransformerM1(input_size=3, dim_val=dv,
                                 window_size=window_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model, best_loss = train_model(
            model, train_loader, optimizer, loss_fn, device,
            n_epoch=ne, tag=f"[{model_name}]",
        )

        model_params["loss"] = [round(best_loss, 5)]
        torch.save(model.state_dict(), "models/" + model_name + ".pt")
        model_params_df = pd.concat(
            [model_params_df, pd.DataFrame(model_params)]
        )
        model_counter += 1

    print("Search done!")
    return model_params_df


def evaluate_search(model_params_df, cfg, data, device, loss_fn):
    valid_df = data["valid_df"]
    valid_phases = data["valid_phases"]
    window_size = cfg["window_size"]
    electron_range = cfg["electron_range"]
    stats = data["stats"]

    results_rows_all = []
    rising_rows_all = []

    # Evaluate each model on a validation loader built for its forecast horizon
    for _, row in model_params_df.iterrows():
        fo = int(row["forecast_out"])
        valid_dataset = ParticleFluxDatasetM1(valid_df, electron_range, fo)
        valid_loader = DataLoader(valid_dataset, batch_size=int(row["batch_size"]))

        one = pd.DataFrame([row])
        res_df, res_rising_df, _ = Evaluate_Models(
            one, model_class=SEPTransformerM1,
            test_loader=valid_loader, test_df=valid_df,
            torres_merged_phases=valid_phases,
            loss_fn=loss_fn, device=device, window_size=window_size,
            input_size=3, p_tot_std=stats["p_tot_std"],
            p_tot_mean=stats["p_tot_mean"], create_graphs=False,
        )
        results_rows_all.append(res_df.iloc[0])
        rising_rows_all.append(res_rising_df.iloc[0])

    results_df = pd.DataFrame(results_rows_all).reset_index(drop=True)
    results_rising_df = pd.DataFrame(rising_rows_all).reset_index(drop=True)

    model_params_df = model_params_df.reset_index(drop=True)
    model_params_df[["o_r2", "o_mse", "o_mae"]] = results_df[["r2", "mse", "mae"]].values
    model_params_df[["r_r2", "r_mse", "r_mae"]] = results_rising_df[["r2", "mse", "mae"]].values

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    model_params_df.to_csv(f"results/model_params_{now}.csv", index=False)
    return model_params_df


# =====================================================================
# FINAL 5-SEED RUNS
# =====================================================================
def run_seed_experiment(model_params_df, cfg, data, device, loss_fn):
    train_df = data["train_df"]
    test_df = data["test_df"]
    torres_merged_phases = data["torres_merged_phases"]
    electron_range = cfg["electron_range"]
    window_size = cfg["window_size"]
    num_workers = cfg["num_workers"]
    stats = data["stats"]

    forecasts = np.unique(model_params_df["forecast_out"].values)
    final_summary = []

    for f in forecasts:
        os.makedirs(f"models/best/t={f}", exist_ok=True)
        for file in os.listdir(f"models/best/t={f}"):
            fp = os.path.join(f"models/best/t={f}", file)
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)

        mpf = model_params_df[model_params_df["forecast_out"] == f].reset_index(drop=True)
        best_index = mpf["r_mae"].idxmin()
        best_model_name = mpf["model_name"].iloc[best_index]
        print(f"Selected t={f}: {best_model_name}, MAE {mpf['r_mae'].min():.3f}")

        dim_value = int(mpf["dim_value"].iloc[best_index])
        batch_size = int(mpf["batch_size"].iloc[best_index])
        learning_rate = mpf["learning_rate"].iloc[best_index]

        seed_test_maes = []
        for seed in range(5):
            torch.manual_seed(seed)

            train_dataset = ParticleFluxDatasetM1(train_df, electron_range, f)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      num_workers=num_workers, pin_memory=True, persistent_workers=True)
            test_dataset = ParticleFluxDatasetM1(test_df, electron_range, f)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, persistent_workers=True)

            model = SEPTransformerM1(input_size=3, dim_val=dim_value,
                                     window_size=window_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model, _ = train_model(
                model, train_loader, optimizer, loss_fn, device,
                n_epoch=1000, tag=f"[t={f} seed{seed}]",
            )

            model_name = f"{best_model_name}-seed{seed}"
            torch.save(model.state_dict(), f"models/best/t={f}/{model_name}.pt")

            one_model_df = pd.DataFrame([{
                "model_name": model_name, "dim_value": dim_value,
                "batch_size": batch_size, "n_epoch": 1000,
                "learning_rate": learning_rate, "forecast_out": f,
            }])

            _, rising_df, _ = Evaluate_Models(
                one_model_df, model_class=SEPTransformerM1,
                test_loader=test_loader, test_df=test_df,
                torres_merged_phases=torres_merged_phases,
                loss_fn=loss_fn, device=device, window_size=window_size,
                input_size=3, p_tot_std=stats["p_tot_std"],
                p_tot_mean=stats["p_tot_mean"],
                models_dir=f"models/best/t={f}/", create_graphs=False,
            )
            seed_mae = rising_df["mae"].iloc[0]
            seed_test_maes.append(seed_mae)
            print(f"t={f} seed {seed} done — rising MAE {seed_mae:.4f}")

        seed_test_maes = np.array(seed_test_maes)
        print(f"\n=== t={f} — Test rising-phase MAE: "
              f"{seed_test_maes.mean():.4f} ± {seed_test_maes.std():.4f} ===\n")
        final_summary.append({
            "forecast_out": f,
            "mean_mae": float(seed_test_maes.mean()),
            "std_mae": float(seed_test_maes.std()),
            "all_maes": list(seed_test_maes),
        })
        # save incrementally so a later crash doesn't lose earlier horizons
        pd.DataFrame(final_summary).to_csv(
            "results/final_seed_summary.csv", index=False
        )

    print("All seed runs done.")
    return final_summary


# =====================================================================
# LINEAR REGRESSION BASELINE
# =====================================================================
def build_flat_dataset(df, electron_range, forecast_out):
    X, y = [], []
    e150 = df["E150"].values
    e300 = df["E300"].values
    p_tot = df["P_tot"].values
    n = len(df) - electron_range - forecast_out
    for idx in range(n):
        window = np.column_stack([
            e150[idx:idx + electron_range],
            e300[idx:idx + electron_range],
            p_tot[idx:idx + electron_range],
        ]).flatten()
        X.append(window)
        y.append(p_tot[idx + electron_range + forecast_out])
    return np.array(X), np.array(y)


def run_linear_baseline(cfg, data, forecast_out=6):
    train_df = data["train_df"]
    test_df = data["test_df"]
    torres_merged_phases = data["torres_merged_phases"]
    electron_range = cfg["electron_range"]
    stats = data["stats"]
    p_tot_std, p_tot_mean = stats["p_tot_std"], stats["p_tot_mean"]

    X_train, y_train = build_flat_dataset(train_df, electron_range, forecast_out)
    X_test, y_test = build_flat_dataset(test_df, electron_range, forecast_out)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    offset = electron_range + forecast_out
    lr_aligned_df = test_df.iloc[offset: offset + len(y_pred)].copy()
    lr_aligned_df["predicted"] = y_pred * p_tot_std + p_tot_mean
    lr_aligned_df["original"] = y_test * p_tot_std + p_tot_mean


    metrics = Get_Rising_Metrics(lr_aligned_df, torres_merged_phases, forecast_out)
    print(f"Linear regression t={forecast_out} rising-phase metrics: {metrics}")

    return metrics

def evaluate_saved_seeds(cfg, data, device, loss_fn):
    test_df = data["test_df"]
    torres_merged_phases = data["torres_merged_phases"]
    electron_range = cfg["electron_range"]
    window_size = cfg["window_size"]
    stats = data["stats"]

    summary_rows = []      # one aggregated row per horizon
    per_seed_rows = []     # one row per individual seed (for inspecting outliers)

    for f in cfg["forecasts_out"]:
        model_dir = f"models/best/t={f}/"
        if not os.path.isdir(model_dir):
            print(f"No saved models for t={f}, skipping")
            continue

        test_dataset = ParticleFluxDatasetM1(test_df, electron_range, f)
        test_loader = DataLoader(test_dataset, batch_size=64)

        seed_metrics = []   # collect the full metric dict from each seed

        for model_file in sorted(os.listdir(model_dir)):
            if not model_file.endswith(".pt"):
                continue
            model_name = model_file[:-3]

            one_model_df = pd.DataFrame([{
                "model_name": model_name,
                "dim_value": cfg["dim_values"][0],
                "batch_size": cfg["batch_sizes"][0],
                "n_epoch": 1000,
                "learning_rate": cfg["learning_rates"][0],
                "forecast_out": f,
            }])

            _, rising_df, _ = Evaluate_Models(
                one_model_df, model_class=SEPTransformerM1,
                test_loader=test_loader, test_df=test_df,
                torres_merged_phases=torres_merged_phases,
                loss_fn=loss_fn, device=device, window_size=window_size,
                input_size=3, p_tot_std=stats["p_tot_std"],
                p_tot_mean=stats["p_tot_mean"],
                models_dir=model_dir, create_graphs=False,
            )

            # rising_df is one row of metrics; grab it as a dict
            m = rising_df.iloc[0].to_dict()
            m["forecast_out"] = f
            m["model_name"] = model_name
            seed_metrics.append(m)
            per_seed_rows.append(m)
            # print(f"t={f} {model_name}: MAE {m['mae']:.4f} | PE {m.get('PE', float('nan')):.4f}")

        # aggregate every numeric metric across the seeds for this horizon
        seed_df = pd.DataFrame(seed_metrics)
        numeric_cols = seed_df.select_dtypes(include=[np.number]).columns

        agg = {"forecast_out": f, "n_seeds": len(seed_df)}
        for col in numeric_cols:
            if col == "forecast_out":
                continue
            agg[f"{col}_mean"] = seed_df[col].mean()
            agg[f"{col}_std"] = seed_df[col].std()
        summary_rows.append(agg)

        print(f"\n=== t={f}: "
              f"MAE {agg['mae_mean']:.4f} ± {agg['mae_std']:.4f} | "
              f"PE {agg.get('PE_mean', float('nan')):.4f} ± {agg.get('PE_std', float('nan')):.4f} ===\n")

    # write both: per-seed detail and aggregated summary
    pd.DataFrame(per_seed_rows).to_csv("results/per_seed_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv("results/final_seed_summary.csv", index=False)
    print("Saved-seed evaluation done.")

def view_torres_phases(data):
    torres = pd.read_csv("data/phases/torres_events_converted.csv")
    flux_data = pd.concat([data["train_df"], data["test_df"]], ignore_index=True)
    flux_data = flux_data.set_index("Timestamp")
    flux_data.index = pd.to_datetime(flux_data.index)

    state = {"i": 0}
    n = len(torres)

    fig, ax = plt.subplots(figsize=(9, 5))

    def draw(i):
        ax.clear()
        row = torres.iloc[i]
        onset = pd.to_datetime(row["Onset Timestamp"])
        peak  = pd.to_datetime(row["Peak Timestamp"])
        end   = pd.to_datetime(row["End Timestamp"])

        idx = flux_data.index.get_indexer([onset, peak, end], method="nearest")
        start = max(0, idx[0] - 5)
        stop  = min(len(flux_data), idx[2] + 5)

        y = flux_data["P_tot"].iloc[start:stop]
        x = np.arange(len(y))
        ax.plot(x, y.values)

        for label, pos in [("onset", idx[0]), ("peak", idx[1]), ("end", idx[2])]:
            rel = pos - start
            if 0 <= rel < len(y):
                ax.axvline(rel, linestyle="--", alpha=0.5)

        ax.set_title(f"Event {i+1}/{n}: peak={row['Peak Flux']:.1f} pfu @ {peak.date()}"
                     f"   (← → to navigate)")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["i"] = (state["i"] + 1) % n
            draw(state["i"])
        elif event.key == "left":
            state["i"] = (state["i"] - 1) % n
            draw(state["i"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(0)
    plt.show()


# =====================================================================
# MAIN
# =====================================================================
def main():


    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    loss_fn = nn.MSELoss()

    # merge_raw_data()   # uncomment to regenerate merged CSVs (run once)

    data = prepare_data(cfg)

    # view_torres_phases(data)


    if cfg["train_new_models"]:
        model_params_df = run_search(cfg, data, device, loss_fn)
        model_params_df = evaluate_search(model_params_df, cfg, data, device, loss_fn)
    else:
        # load most recent search results
        files = sorted(f for f in os.listdir("results/")
                       if f.startswith("model_params_"))
        model_params_df = pd.read_csv(os.path.join("results/", files[-1]))


    if cfg["seed_experiment"]:
        run_seed_experiment(model_params_df, cfg, data, device, loss_fn)
    else:
        evaluate_saved_seeds(cfg, data, device, loss_fn)

    all_t_metrics = {}
    for f in cfg["forecasts_out"]:
        
        all_t_metrics["t=" + str(f)] = run_linear_baseline(cfg, data, forecast_out=f)

    # Convert nested dict to a dataframe: one row per horizon, one column per metric
    baseline_df = pd.DataFrame(all_t_metrics).T   # .T so horizons are rows, metrics are columns
    baseline_df.index.name = "horizon"
    baseline_df = baseline_df.reset_index()

    baseline_df.to_csv("results/linear_baseline_metrics.csv", index=False)
    print(baseline_df.to_string())
    

if __name__ == "__main__":
    main()