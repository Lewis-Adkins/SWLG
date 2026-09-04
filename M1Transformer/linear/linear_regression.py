import os
from sklearn.linear_model import LinearRegression
import numpy as np
from torres.stats import mae, x_axis_error, lag_ln10
from torres.m1 import evaluate

def run_linear_baseline(train, targets_train, test, targets_test, data, prediction_time,
                        result_path, event_times):
    """Fit + evaluate the sklearn LinearRegression baseline on one already-built
    dataset variant (from pair_input_output's n_datasets output) and write its
    predictions/results into the standard per-seed result directory, exactly
    where a transformer seed M1-00 would write them:

        results/{base}/resutls_per_dataset/t+{pt}/{run_tag}/dataset{j}/M1-00/

    so score_forecast, create_result_csv and the plotting code all pick it up
    through the same code path as sin/rope/zero -- no linear-specific
    aggregation functions. `base` is linear/{split_type}/{phases_dir} via
    _result_base (model_type == "linear"), `run_tag` is linear_{phases}.

    :param result_path: the M1-00 directory to write predictions.txt/results.txt into
    :param event_times: parsed data/event_timestamps.txt rows (list of [onset, ...])
    """

    X_train = np.array([instance.flatten() for instance in train])
    X_test = np.array([instance.flatten() for instance in test])

    lr_model = LinearRegression()
    lr_model.fit(X_train, targets_train)
    y_pred = lr_model.predict(X_test)

    os.makedirs(result_path, exist_ok=True)
    np.savetxt(f"{result_path}/predictions.txt", y_pred, delimiter=",")

    # Only events whose full onset..peak span is present in this variant's
    # test set (works for the real dataset and any block-partitioned variant).
    event_times_test = [event for event in event_times if event[0] in targets_test]

    target_times = list(targets_test.keys())
    predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}

    evaluate(targets_test, predictions, event_times_test, data, path=result_path, display=False)

    return y_pred, targets_test
