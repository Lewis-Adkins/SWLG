import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torres.m1 import _partition_windows, evaluate


def _build_posner_windows(data, prediction_time):
    """(intensity, log_max_rise) feature pairs + targets, reproduced from
    torres/posner_method.py's original pair_input_output/main (Posner 2007).
    Kept pixel-identical to that file's math, INCLUDING its one-step-ahead
    read of electron[t+1] in the last max_rise term (traced and confirmed
    intentional with the project owner, not "fixed" here -- see notes/debug).

    electron_t is intensity at time t (clipped to [-3, 8]); max_rise is the
    largest 5-minute slope of electron flux over the 13 points from t-12 to
    t+1 (clipped to [1e-2, 0.2], then log'd). Same t-range as
    torres.m1._build_windows (`range(24, len(data) - prediction_time)`) so
    target_rows lines up exactly with the transformer/linear/persistence
    windows built from the same (data, prediction_time) -- feeding both into
    the same _partition_windows gives every model type IDENTICAL train/test
    row splits per dataset variant."""
    time = list(data['time'].values)
    electron = data['electron'].values
    proton = data['proton'].values

    x = []
    y = []
    target_rows = []

    for t in range(24, len(data) - prediction_time):
        max_rise = (electron[t - 12] - electron[t - 11]) / 5
        for interval in range(11, -1, -1):
            max_rise = max([max_rise, (electron[t - interval] - electron[t - interval + 1]) / 5])

        if max_rise <= 1e-2:
            max_rise = 1e-2
        elif max_rise >= 0.2:
            max_rise = 0.2

        if electron[t] > 8:
            electron_t = 8
        elif electron[t] < -3:
            electron_t = -3
        else:
            electron_t = electron[t]

        x.append([electron_t, np.log(max_rise)])
        y.append([time[t + prediction_time], proton[t + prediction_time]])
        target_rows.append(t + prediction_time)

    return x, y, np.array(target_rows)


def pair_input_output(data, prediction_time, n_datasets=1, train_split=0.8,
                      size_blocks=6000, random_state=42,
                      event_path="data/event_indices.txt"):
    """Posner-feature equivalent of torres.m1.pair_input_output: same
    n_datasets bootstrap partitioning (dataset 0 = real chronological split,
    1..n-1 = event-safe random block partitions), via the shared
    torres.m1._partition_windows -- see _build_posner_windows for why this
    produces the same row splits as the transformer/linear/persistence
    windows for a given (data, prediction_time, n_datasets, ...)."""
    x, y, target_rows = _build_posner_windows(data, prediction_time)
    return _partition_windows(x, y, target_rows, len(data), n_datasets, train_split,
                              size_blocks, random_state, event_path)


def _train_grid(train, targets_train, prediction_time, path):
    """Fit the Posner forecasting matrix: bin training instances into an
    18 (intensity) x 13 (slope) grid and average each cell's target proton
    flux. Ported near-verbatim from torres/posner_method.py's train_model."""
    min_intensity = np.min(train[:, 0])
    max_intensity = np.max(train[:, 0])
    intensity_ranges = np.linspace(min_intensity, max_intensity, 19)
    min_slope = np.min(train[:, 1])
    max_slope = np.max(train[:, 1])
    slope_ranges = np.linspace(min_slope, max_slope, 14)

    range_matrix = [[0 for _ in range(len(slope_ranges) - 1)] for _ in range(len(intensity_ranges) - 1)]
    for i in range(len(intensity_ranges) - 1):
        for j in range(len(slope_ranges) - 1):
            range_matrix[i][j] = [(intensity_ranges[i], intensity_ranges[i + 1]),
                                  (slope_ranges[j], slope_ranges[j + 1])]
    range_matrix = np.flipud(range_matrix)  # match Posner paper's plotting orientation

    if path:
        plt.hist2d(train[:, 1], train[:, 0], bins=[slope_ranges, intensity_ranges], norm=matplotlib.colors.LogNorm())
        plt.xlabel("Slope")
        plt.ylabel("Intensity")
        plt.colorbar()
        plt.savefig(f"{path}/posner_instances_per_cell_t+{prediction_time}.png")
        plt.close()

    model = [[0 for _ in range(range_matrix.shape[1])] for _ in range(range_matrix.shape[0])]
    num_instances_per_cell = [[0 for _ in range(range_matrix.shape[1])] for _ in range(range_matrix.shape[0])]
    for train_i, targets_train_i in zip(train, targets_train):
        matched = False

        if train_i[0] == max_intensity:
            for j in range(range_matrix.shape[1]):
                if range_matrix[0][j][1][0] <= train_i[1] < range_matrix[0][j][1][1]:
                    model[0][j] += targets_train_i
                    num_instances_per_cell[0][j] += 1
                    matched = True
                    break

            if train_i[1] == max_slope:
                model[0][-1] += targets_train_i
                num_instances_per_cell[0][-1] += 1
                continue

        for i in range(range_matrix.shape[0]):
            if matched:
                break
            for j in range(range_matrix.shape[1]):
                if range_matrix[i][j][0][0] <= train_i[0] < range_matrix[i][j][0][1] and \
                        range_matrix[i][j][1][0] <= train_i[1] < range_matrix[i][j][1][1]:
                    model[i][j] += targets_train_i
                    num_instances_per_cell[i][j] += 1
                    matched = True
                    break

            if train_i[1] == max_slope and range_matrix[i][-1][0][0] <= train_i[0] < range_matrix[i][-1][0][1]:
                model[i][-1] += targets_train_i
                num_instances_per_cell[i][-1] += 1
                break

    model = np.array(model, dtype=np.float64)
    num_instances_per_cell = np.array(num_instances_per_cell)
    # Empty cells (no training instance ever fell in them) divide 0/0 -> nan;
    # left as nan (not the original's silent RuntimeWarning-and-nan) since
    # _predict_grid below explicitly falls back to the global mean for them.
    with np.errstate(invalid="ignore"):
        model /= num_instances_per_cell

    if path:
        plt.imshow(model)
        plt.title("Forecasting Matrix")
        plt.xlabel("Electron rise")
        xtick_labels = [f"{(slope_ranges[i] + slope_ranges[i + 1]) / 2:.02f}" for i in range(13)]
        plt.xticks(np.arange(13), xtick_labels)
        plt.ylabel("Electron intensity")
        ytick_labels = [f"{(intensity_ranges[i] + intensity_ranges[i + 1]) / 2:.02f}" for i in range(17, -1, -1)]
        plt.yticks(np.arange(18), ytick_labels)
        plt.colorbar()
        plt.savefig(f"{path}/posner_model_t+{prediction_time}.png")
        plt.close()

    return model, range_matrix


def _predict_grid(test, model, range_matrix, fallback_value):
    """Look up each test instance's cell in the fitted grid. Ported from
    torres/posner_method.py's predict, with one addition: a test instance
    whose cell had zero training instances (nan in `model`) falls back to
    `fallback_value` (the training-set mean) instead of silently predicting
    nan straight into evaluate()'s MAE/lag math -- the original script never
    exercised this path since it only ever ran on one fixed 80/20 split."""
    min_intensity = range_matrix[-1][0][0][0]
    max_intensity = range_matrix[0][0][0][1]
    min_slope = range_matrix[0][0][1][0]
    max_slope = range_matrix[0][-1][1][1]

    predictions = []
    for test_i in test:
        test_i = test_i.copy()
        if test_i[0] < min_intensity:
            test_i[0] = min_intensity
        elif test_i[0] > max_intensity:
            test_i[0] = max_intensity
        if test_i[1] < min_slope:
            test_i[1] = min_slope
        elif test_i[1] > max_slope:
            test_i[1] = max_slope

        matched = False
        value = None
        if test_i[0] == max_intensity:
            for j in range(range_matrix.shape[1]):
                if range_matrix[0][j][1][0] <= test_i[1] < range_matrix[0][j][1][1]:
                    value = model[0][j]
                    matched = True
                    break

            if test_i[1] == max_slope:
                value = model[0][-1]
                matched = True

        if not matched:
            for i in range(range_matrix.shape[0]):
                if matched:
                    break
                for j in range(range_matrix.shape[1]):
                    if range_matrix[i][j][0][0] <= test_i[0] < range_matrix[i][j][0][1] and \
                            range_matrix[i][j][1][0] <= test_i[1] < range_matrix[i][j][1][1]:
                        value = model[i][j]
                        matched = True
                        break

                if test_i[1] == max_slope and range_matrix[i][-1][0][0] <= test_i[0] < range_matrix[i][-1][0][1]:
                    value = model[i][-1]
                    matched = True
                    break

        if value is None or np.isnan(value):
            value = fallback_value
        predictions.append(value)

    return np.array(predictions)


def run_posner_baseline(train, targets_train, test, targets_test, data, prediction_time,
                        result_path, event_times):
    """Fit + evaluate the Posner (2007) lookup-table baseline on one already-
    partitioned dataset variant (torres/posner_method.pair_input_output's
    n_datasets output) and write into the standard per-seed result directory
    -- results/{base}/resutls_per_dataset/t+{pt}/{run_tag}/dataset{j}/M1-00/
    -- exactly like linear/persistence, so score_forecast/create_result_csv
    need no posner-specific branch.
    """
    train = np.array(train, dtype=np.float64)
    test = np.array(test, dtype=np.float64)

    os.makedirs(result_path, exist_ok=True)
    model, range_matrix = _train_grid(train, targets_train, prediction_time, result_path)
    y_pred = _predict_grid(test, model, range_matrix, fallback_value=np.mean(targets_train))

    np.savetxt(f"{result_path}/predictions.txt", y_pred, delimiter=",")

    event_times_test = [event for event in event_times if event[0] in targets_test]
    target_times = list(targets_test.keys())
    predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}

    evaluate(targets_test, predictions, event_times_test, data, path=result_path, display=False)

    return y_pred, targets_test
