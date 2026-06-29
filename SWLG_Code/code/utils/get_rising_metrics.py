from utils.slice_rising import Slice_Rising
import sklearn as sk
import numpy as np
import pandas as pd

def compute_lag(rising_df, pred_col, max_shift=12):
    """Find the shift minimizing MAE between `pred_col` and original, per event. Mean across events."""
    lags = []
    for event_id, event in rising_df.groupby("event_id"):
        target = event["original"].values
        pred = event[pred_col].values

        if len(target) < 2:
            continue

        best_shift, best_mae = 0, np.inf
        for shift in range(0, min(max_shift, len(target))):
            if shift == 0:
                shifted_pred, t = pred, target
            else:
                shifted_pred, t = pred[:-shift], target[shift:]
            if len(t) == 0:
                break
            mae = np.mean(np.abs(t - shifted_pred))
            if mae < best_mae:
                best_mae, best_shift = mae, shift
        lags.append(best_shift)
    return np.mean(lags) if lags else np.nan

def Get_Rising_Metrics(a_df, a_phase_df, a_future_proton_out):
    # Compute persistence on the CONTINUOUS series first:
    # prediction for t+h is the actual value at t  ->  shift the full series by h
    a_df = a_df.copy()
    a_df["persistence"] = a_df["original"].shift(a_future_proton_out)

    rising_df = Slice_Rising(a_df, a_phase_df)

    # --- ALIGNMENT CHECK: does predicted track original within an event? ---
    for eid in rising_df["event_id"].unique()[:3]:   # first 3 events
        ev = rising_df[rising_df["event_id"] == eid]
        print(f"\n--- event {eid} ({len(ev)} rows) ---", flush=True)
        print(ev[["Timestamp", "original", "predicted"]].to_string(), flush=True)
    # --- END CHECK ---

    if rising_df.empty:
        print("WARNING: no rising-phase events found in this data window — returning NaN metrics", flush=True)
        nan = float("nan")
        return {k: nan for k in ["mae", "persist_mae", "r2", "mse", "model_lag", "persist_lag", "predictve_efficiency"]}

    # Drop only rows where persistence is undefined (start of the whole series)
    valid = rising_df["persistence"].notna()


#    # --- DIAGNOSTIC ---
#     print("=== rising_df after slice ===")
#     print(rising_df[["Timestamp", "original", "predicted", "persistence"]].head(20).to_string())
#     print("original range:", rising_df["original"].min(), "to", rising_df["original"].max())
#     print("predicted range:", rising_df["predicted"].min(), "to", rising_df["predicted"].max())
#     print("rows:", len(rising_df), "events:", rising_df["event_id"].nunique())
#     # --- END DIAGNOSTIC ---


    mae_rising  = sk.metrics.mean_absolute_error(rising_df['original'][valid], rising_df['predicted'][valid])
    mae_persist = sk.metrics.mean_absolute_error(rising_df['original'][valid], rising_df['persistence'][valid])
    r2_rising   = sk.metrics.r2_score(rising_df['original'][valid], rising_df['predicted'][valid])
    mse_rising  = sk.metrics.mean_squared_error(rising_df['original'][valid], rising_df['predicted'][valid])
    predictve_efficiency = 1 - mse_rising/np.var(rising_df['original'][valid])
    
    metrics = {
        "mae": mae_rising,
        "persist_mae": mae_persist,
        "r2": r2_rising,
        "mse": mse_rising,
        "model_lag": compute_lag(rising_df, "predicted"),
        "persist_lag": compute_lag(rising_df, "persistence"),
        "predictve_efficiency": predictve_efficiency
    }
    df_final_seed_metrics = pd.DataFrame(metrics, index = [0])



    #df_final_seed_metrics.to_csv("results/final.csv")
    return metrics