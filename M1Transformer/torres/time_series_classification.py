from torres.load_data import *
from torres.m1 import pair_input_output
import numpy as np
import pandas as pd
import sys

def _count_tp_fp_fn(predictions, targets_train, prediction_time, app, alert_window, event_times_test):
    """
    Count TPs, FPs, and FNs for a single set of predictions at a given alert window.
    :param predictions: List/array of predicted proton flux values for the test set
    :param targets_train: The training targets (only its length is used, for offsetting indices)
    :param prediction_time: The number of timesteps ahead being predicted
    :param app: Alerting approach ("app0" - "app3")
    :param alert_window: The alert window size, in timesteps
    :param event_times_test: List of [threshold, end] index pairs for SEP events in the test set
    :return: (tp, fp, fn)
    """

    predictions = np.array(predictions)

    # Collect list of alert durations (start index, end index)
    alert_durations = []
    t = 0
    while t < len(predictions):

        # Start alert duration
        if predictions[t] >= np.log(10):
            start = t + 24 + len(targets_train) + prediction_time

            # Blue bar
            if app in ["app2", "app3"]:
                start -= prediction_time

            # Extend alert duration as needed, starting from last timestamp with intensity > ln10 (yellow/green bar)
            if app in ["app1", "app2"]:
                while True in (predictions[t + 1: t + alert_window + 1] > np.log(10)):
                    for t2 in range(alert_window, 0, -1):
                        if t + t2 < len(predictions) and predictions[t + t2] >= np.log(10):
                            t += t2
                            break
            else:

                # Yellow bar without green bar
                while t + 1 < len(predictions) and predictions[t + 1] >= np.log(10):
                    t += 1

            end = t + alert_window + 24 + len(targets_train) + prediction_time  # 6 hours after t, offset by training set
            alert_durations.append([start, end])
            t += alert_window
            if app in ["app2", "app3"]:
                t += prediction_time  # Prevent overlapping alerts

        else:
            t += 1

    # Check for TPs and FPs - for each alert duration, does it overlap the start of an SEP event,
    # is there no SEP event for the entire alert duration, or does the alert take place too late?
    tp = 0
    fp = 0
    fn = 0

    for j, event_duration in enumerate(event_times_test):

        overlap = False
        for alert_duration in alert_durations:
            if alert_duration[0] < event_duration[0] < alert_duration[1]:
                overlap = True
                tp += 1
                break

        if not overlap:
            fn += 1

    for alert_duration in alert_durations:

        # Check for overlap with any SEP event, and determine which case
        overlap = False
        for event_duration in event_times_test:

            # Alert duration overlaps beginning of SEP event - TP, but it was already counted so just ignore
            if alert_duration[0] < event_duration[0] < alert_duration[1]:
                overlap = True
                break

            # Alert duration is entirely inside SEP event - not counted
            elif event_duration[0] < alert_duration[0] < alert_duration[1] < event_duration[1]:
                overlap = True
                break

            # Overlapping with end of event but not beginning; not counted
            elif event_duration[0] < alert_duration[0] < event_duration[1] < alert_duration[1]:
                overlap = True
                break

        # If alert duration does not overlap any SEP event, it is an FP
        if not overlap:
            fp += 1

    return tp, fp, fn


def _test_setup(prediction_time):
    """
    Shared setup for scoring functions: pairs inputs/outputs and loads SEP events
    that fall within the test set.

    Only ever uses dataset 0 (the real, untouched chronological split) --
    this whole scoring path is positional (row-offset math tied to one
    contiguous test tail), which doesn't hold for the block-partitioned
    dataset variants pair_input_output can also produce; see main.py, where
    score_forecast/score_forecast_linear are only called for dataset 0.
    :param prediction_time: The number of timesteps ahead being predicted
    :return: (targets_train, event_times_test)
    """

    data = pd.read_csv("data/data.csv")
    trains, targets_trains, tests, targets_tests = pair_input_output(data, False, prediction_time)
    targets_train = targets_trains[0]

    # Load events for evaluation
    event_file = open('data/event_indices.txt', 'r')
    lines = event_file.readlines()
    event_times = [line.split() for line in lines]
    event_times = [[int(index) for index in event] for event in event_times]
    event_file.close()

    # Select only events which fall in the test set
    event_times_test = []
    for event in event_times:
        if event[0] >= 24 + len(targets_train):
            event_times_test.append([event[1], event[3]])  # Threshold, end

    return targets_train, event_times_test


def score_forecast(prediction_time, app, run_tag, n_seeds=5):
    # Only ever scores dataset 0 -- see _test_setup's docstring.
    path = f"results/transformer/t+{prediction_time}/{run_tag}/dataset0"
    # app0 - approach W
    # app1 - approach EW
    # app2 - approach EAW
    # app3 - approach AW
    if app not in [f"app{i}" for i in range(4)]:
        print("Invalid app selection, exiting.")
        exit(0)

    targets_train, event_times_test = _test_setup(prediction_time)

    records = []

    # Evaluate for different alert windows
    for alert_window in [1, 3, 6, 9, 12, 24, 72]:

        if alert_window < 12:
            print(f"Alert window = {alert_window * 5} minutes")
        elif alert_window == 12:
            print("Alert window = 1 hour")
        else:
            print(f"Alert window = {alert_window * 5 // 60} hours")

        tps = []
        fps = []
        fns = []

        # Evaluate predictions from each run
        for i in range(1, n_seeds + 1):
            predictions = load_series(f"{path}/M1-{str(i-1).zfill(2)}/predictions.txt")

            tp, fp, fn = _count_tp_fp_fn(predictions, targets_train, prediction_time, app, alert_window, event_times_test)

            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            print(f"Run {i}: TP = {tp}, FN = {fn}, FP = {fp}")
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            records.append({
                "t": prediction_time,
                "app": app,
                "alert_window": alert_window,
                "model": f"M1-{str(i - 1).zfill(2)}",
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "F1": round(f1, 3),
            })
        tps = np.array(tps)
        fps = np.array(fps)
        fns = np.array(fns)
        f1s = (2 * tps) / (2 * tps + fps + fns)
        print(f"Average: TP = {np.mean(tps):.2f}, FN = {np.mean(fns):.2f}, FP = {np.mean(fps):.2f}, F1 = {np.mean(f1s):.2f}\n", )

        # If not using green bar, no need to loop over remaining alert window sizes
        # if alert_window == 1 and app in ["app0", "app3"]:
        #     exit(0)
    return records


def score_forecast_linear(prediction_time, app, run_tag):
    """
    Same alert-window F1 scoring as score_forecast, but for the linear regression
    baseline, which has a single run (no seeds) at results/linear/t+{prediction_time}/{run_tag}.
    """

    # Only ever scores dataset 0 -- see _test_setup's docstring.
    path = f"results/linear/t+{prediction_time}/{run_tag}/dataset0"
    if app not in [f"app{i}" for i in range(4)]:
        print("Invalid app selection, exiting.")
        exit(0)

    targets_train, event_times_test = _test_setup(prediction_time)

    predictions = load_series(f"{path}/predictions.txt")

    records = []

    for alert_window in [1, 3, 6, 9, 12, 24, 72]:

        if alert_window < 12:
            print(f"Alert window = {alert_window * 5} minutes")
        elif alert_window == 12:
            print("Alert window = 1 hour")
        else:
            print(f"Alert window = {alert_window * 5 // 60} hours")

        tp, fp, fn = _count_tp_fp_fn(predictions, targets_train, prediction_time, app, alert_window, event_times_test)

        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        print(f"Linear: TP = {tp}, FN = {fn}, FP = {fp}, F1 = {f1:.2f}\n")
        records.append({
            "t": prediction_time,
            "app": app,
            "alert_window": alert_window,
            "model": "linear",
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "F1": round(f1, 3),
        })

    return records
