# utils/model.py
import torch
import torch as tc
import pandas as pd

from utils.graph_results import Record_Overall_Results
from utils.get_rising_metrics import Get_Rising_Metrics

from datetime import datetime
def Evaluate_Models(
    model_params_df,
    *,
    model_class,            # e.g. SEPTransformerM1
    test_loader,
    test_df,
    torres_merged_phases,
    loss_fn,                # the MSE callable
    device,
    window_size,
    input_size,
    p_tot_std,
    p_tot_mean,
    models_dir="models/",
    create_graphs,
):
    """Evaluate each trained model in model_params_df on the test set.

    Inference only — no gradients, no optimizer, no parameter updates.
    This is NOT training despite any caller alias.

    Returns
    -------
    results_df : DataFrame        # predicted-panel r2/mse/mae per model
    results_rising_df : DataFrame # Get_Rising_Metrics output per model
    per_model_outputs : dict      # model_name -> {aligned_df, losses, test_loss}
    """
    results_rows = []
    rising_rows = []
    per_model_outputs = {}



    for _, row in model_params_df.iterrows():
        model = model_class(
            input_size=input_size,
            dim_val=row["dim_value"],
            window_size=window_size,
        ).to(device)

        offset = window_size + row["forecast_out"]
        model.load_state_dict(
            tc.load(models_dir + row["model_name"]  + '.pt', map_location=device)
        )
        model.eval()

        predictions = tc.tensor([]).to(device)
        original    = tc.tensor([]).to(device)
        losses      = tc.tensor([]).to(device)
        test_loss   = 0.0

        with torch.no_grad():
            for batch in test_loader:
                x = batch["Current_Flux_Data"].to(device)
                y = batch["Target_Proton_Data"].to(device)

                pred = model(x).squeeze()
                original    = tc.cat((original, y.flatten()))
                predictions = tc.cat((predictions, pred.flatten()))

                loss = loss_fn(y, pred)
                losses = tc.cat((losses, loss.flatten()))
                test_loss += loss.item()

        pred_log = predictions.detach().cpu().numpy() * p_tot_std + p_tot_mean
        orig_log = original.detach().cpu().numpy() * p_tot_std + p_tot_mean

        test_aligned_df = test_df.iloc[offset : offset + len(predictions)].copy()
        test_aligned_df["predicted"] = pred_log
        test_aligned_df["original"]  = orig_log



        # --- rising-phase metrics ---
        rising_results = Get_Rising_Metrics(
            test_aligned_df, torres_merged_phases, row["forecast_out"]
        )
        rising_results["model_name"] = row["model_name"]
        rising_rows.append(rising_results)

        # --- overall metrics + plot (writes results/plots/<model_name>.png) ---
        results = Record_Overall_Results(
            test_aligned_df, row["batch_size"], losses, row["model_name"], create_graphs
        )
        results["model_name"] = row["model_name"]
        results_rows.append(results)

        per_model_outputs[row["model_name"]] = {
            "aligned_df": test_aligned_df,
            "losses": losses.detach().cpu().numpy(),
            "test_loss": test_loss,
        }

    results_df        = pd.DataFrame(results_rows).reset_index(drop=True)
    results_rising_df = pd.DataFrame(rising_rows).reset_index(drop=True)



    return results_df, results_rising_df, per_model_outputs