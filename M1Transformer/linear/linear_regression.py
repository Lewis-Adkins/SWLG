import os
from sklearn.linear_model import LinearRegression
import numpy as np
from torres.stats import mae, x_axis_error, lag_ln10
from torres.m1 import evaluate

def run_linear_baseline(train, targets_train, test, targets_test, data, prediction_time, run_tag, dataset_id=0):
    """Train and evaluate the linear baseline on an already-built dataset
    variant (from pair_input_output's n_datasets output) rather than calling
    pair_input_output itself, so it shares the exact same dataset variants
    the transformer models train on."""

    X_train = np.array([instance.flatten() for instance in train])
    X_test = np.array([instance.flatten() for instance in test])

    lr_model = LinearRegression()
    lr_model.fit(X_train, targets_train)
    y_pred = lr_model.predict(X_test)

    path = f"results/linear/t+{prediction_time}/{run_tag}/dataset{dataset_id}"
    os.makedirs(path, exist_ok=True)
    np.savetxt(f"{path}/predictions.txt", y_pred, delimiter=",")

    # Load events for evaluation
    event_file = open('data/event_timestamps.txt', 'r')
    lines = event_file.readlines()
    event_times = [line.split() for line in lines]
    event_file.close()

    # Only events whose full onset..peak span is present in this variant's
    # test set (works for the real dataset and any block-partitioned variant).
    event_times_test = [event for event in event_times if event[0] in targets_test]

    # Evaluate predictions
    target_times = list(targets_test.keys())
    predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}

    evaluate(targets_test, predictions, event_times_test, data, path = path, display= False)

    return y_pred, targets_test
