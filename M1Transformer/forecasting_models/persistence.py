import os
import numpy as np
from torres.m1 import evaluate

def run_persistence_baseline(test, targets_test, data, prediction_time, result_path, event_times):
    """The naive 'nothing changes' baseline: predict the flux at t+prediction_time
    as whatever it currently is at t. No fitting -- only needs the test
    windows, not train/targets_train.

    Each test window instance is (features, window_size) with channel order
    [electron, electron_high, proton, (phase columns)] and the window's last
    column (index -1) is the current timestep t (see torres/m1.py's
    _build_windows: `proton[t - 24: t + 1]`) -- so instance[2, -1] is exactly
    the current proton flux, regardless of whether this is dataset 0's
    contiguous test tail or a block-partitioned variant's scattered windows.

    Writes into the same M1-00 result directory a one-seed transformer run
    would (see main.py), so score_forecast/create_result_csv need no
    persistence-specific branch.
    """

    PROTON_CHANNEL = 2
    y_pred = np.array([instance[PROTON_CHANNEL, -1] for instance in test])

    os.makedirs(result_path, exist_ok=True)
    np.savetxt(f"{result_path}/predictions.txt", y_pred, delimiter=",")

    # Only events whose full onset..peak span is present in this variant's
    # test set (works for the real dataset and any block-partitioned variant).
    event_times_test = [event for event in event_times if event[0] in targets_test]

    target_times = list(targets_test.keys())
    predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}

    evaluate(targets_test, predictions, event_times_test, data, path=result_path, display=False)

    return y_pred, targets_test
