# from keras.callbacks import EarlyStopping
# from keras.layers import Dense, GRU
# from keras.models import load_model, Sequential
# from load_data import *
from torres.stats import mae, x_axis_error, pe, lag_ln10, tss_f1
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
# from stats import mae
# from stats import lag_ln10
# from stats import x_axis_error

def slice_dictionary_keys(dictionary, start, end):
    """
    Slice a dictionary by keys rather than indices.
    :param dictionary: A dictionary object
    :param start: The starting key
    :param end: The ending key
    :return: The dictionary keys and values between the start and end keys
    """
    times = list(dictionary.keys())
    keys = times[times.index(start): times.index(end) + 1]
    return {key: dictionary[key] for key in keys}


def _event_spans(event_path="data/event_indices.txt", margin=36):
    """(start, end) row-index span for every labeled event, extended `margin`
    rows before onset (matches evaluate()'s own 36-row background lookback)
    through the event's `end` row -- the widest span anything downstream
    indexes into."""
    with open(event_path) as f:
        rows = [[int(v) for v in line.split()] for line in f if line.strip()]
    return [(row[0] - margin, row[3]) for row in rows]


def _event_safe_block_bounds(n, size_blocks, spans):
    """Non-overlapping (start, end) block boundaries tiling [0, n), each
    ~size_blocks rows, with boundaries pushed forward so none ever falls
    inside a labeled event's span -- every event ends up fully inside
    exactly one block, never split across two."""
    bounds = []
    start = 0
    while start < n:
        end = min(start + size_blocks, n)
        grew = True
        while grew:
            grew = False
            for s, e in spans:
                if s < end < e:
                    end = min(e, n)
                    grew = True
        bounds.append((start, end))
        start = end
    return bounds


def _build_windows(data, use_phase_inputs, prediction_time):
    """The pure windowing pass: slide a 24-step lookback across the whole
    (real, untouched, chronologically-ordered) `data` and pair each window
    with its target. Returns (x, y, target_rows) where target_rows[i] is the
    raw row index of x[i]/y[i]'s target (t + prediction_time) -- used to
    decide which train/test block a window belongs to, since the target is
    what has to land on one side or the other of a block boundary intact."""
    time = list(data['time'].values)
    electron = data['electron'].values
    electron_high = data['electron_high'].values
    proton = data['proton'].values
    phases = np.array([data['background'].values, data['rising'].values, data['falling'].values]).T

    x = []
    y = []
    target_rows = []

    for t in range(24, len(data) - prediction_time):

        # Get current instance (past 2 hours plus current)
        x_curr = [electron[t - 24: t + 1], electron_high[t - 24: t + 1], proton[t - 24: t + 1]]

        # Adjust phase inputs by checking if there was a change in phase before the past 30 minutes
        # If so, then starting from last index, go backwards and replace up to t-6 with the value of t-6
        if use_phase_inputs:
            phases_t = np.array(phases[t - 24: t + 1])
            i = 24  # 24 due to size of temporary phase array (instead of directly modifying entire phase data)
            while (phases_t[i] != phases_t[i - 6]).any():
                phases_t[i] = phases_t[i - 6]
                i -= 1

            # Add columns to current instance
            for j in range(phases_t.shape[1]):
                x_curr.append(phases_t[:, j])

        # Add to list of all instances
        x.append(x_curr)

        # Store output; include timestamps
        y.append([time[t + prediction_time], proton[t + prediction_time]])
        target_rows.append(t + prediction_time)

    return x, y, np.array(target_rows)


def pair_input_output(data, use_phase_inputs, prediction_time, n_datasets=1,
                       train_split=0.8, size_blocks=6000, random_state=42,
                       event_path="data/event_indices.txt"):
    """
    Pair inputs and outputs, and split into training and testing sets.

    Returns `n_datasets` versions of each (see below), NOT a single dataset.
    Dataset 0 is always the real, untouched chronological train_split cut --
    identical to this function's pre-`n_datasets` behavior. Datasets
    1..n_datasets-1 are built by randomly relabeling contiguous, event-safe
    blocks as train/test (same real rows, same real order, just a different
    random partition each time -- no row is ever duplicated or dropped; see
    notes/debug for why this is not block-bootstrap-with-replacement).

    :param data: A DataFrame object (never modified or reordered)
    :param use_phase_inputs: Whether or not to use phase inputs
    :param prediction_time: The number of timesteps to predict ahead
    :param n_datasets: How many train/test dataset variants to produce
    :param train_split: Fraction of rows labeled train in each variant
    :param size_blocks: Nominal block size for the random-partition variants
        (datasets 1..n_datasets-1 only; ignored for dataset 0)
    :return: trains, targets_trains, tests, targets_tests -- each a list of
        length n_datasets
    """
    x, y, target_rows = _build_windows(data, use_phase_inputs, prediction_time)
    n_windows = len(x)

    # Dataset 0: the real, untouched chronological split (unchanged from
    # this function's original single-dataset behavior).
    cut = int(train_split * n_windows)
    trains = [np.array(x[:cut])]
    targets_trains = [np.array(y[:cut])[:, 1].astype(np.float64)]
    tests = [np.array(x[cut:])]
    targets_tests = [{t[0]: t[1] for t in y[cut:]}]

    if n_datasets > 1:
        spans = _event_spans(event_path)
        bounds = _event_safe_block_bounds(len(data), size_blocks, spans)
        block_starts = np.array([b[0] for b in bounds])
        row_counts = np.array([b[1] - b[0] for b in bounds])
        # which block each window's target row falls in -- block membership
        # is decided by the target, not the input window's last row, so an
        # event-safe block (guaranteed to contain a whole event's onset..end
        # span) puts every window whose target is part of that event on the
        # same side of the train/test line.
        window_block = np.searchsorted(block_starts, target_rows, side="right") - 1

        for j in range(1, n_datasets):
            rng = np.random.RandomState(random_state + j)
            order = rng.permutation(len(bounds))
            cum_rows = np.cumsum(row_counts[order])
            n_train_target = train_split * len(data)
            n_train_blocks = int(np.searchsorted(cum_rows, n_train_target)) + 1
            train_blocks = set(order[:n_train_blocks].tolist())

            is_train = np.isin(window_block, list(train_blocks))
            train_idx = np.nonzero(is_train)[0]
            test_idx = np.nonzero(~is_train)[0]

            trains.append(np.array([x[i] for i in train_idx]))
            targets_trains.append(np.array([y[i][1] for i in train_idx]).astype(np.float64))
            tests.append(np.array([x[i] for i in test_idx]))
            targets_tests.append({y[i][0]: y[i][1] for i in test_idx})

    return trains, targets_trains, tests, targets_tests


# def train_model(train, targets_train, algorithm):
#     """
#     Create and train the model.
#     :param train: A numpy array in which each row is an instance
#     :param targets_train: The targets for each training instance
#     :param algorithm: 'regular' or 'rnn'
#     :return: The trained model
#     """
#     model = Sequential()
#     if algorithm == 'regular':
#         model.add(Dense(30, input_shape=train.shape[1:], activation='sigmoid'))
#     else:
#         model.add(GRU(30, input_shape=train.shape[1:], activation='sigmoid', return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss='mse', optimizer='adam')
#     model.fit(train, targets_train, epochs=1000, verbose=1,
#               callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
#     return model


def evaluate(targets_test, predictions, event_times, data, path, display):
    # event_times must already be filtered to events whose full onset..peak
    # span is present in targets_test's keys -- callers should filter with
    # `event[0] in targets_test` (works for both the real dataset and any
    # block-partitioned variant, since it checks presence, not position).

    # First, get electron, high energy electron, and xray from data with timestamps for plots
    times = list(data["time"].values)

    # Keyed by timestamp over the *whole* dataset, not sliced by position --
    # targets_test's rows may be a scattered subset (block-partitioned
    # variants), not a contiguous tail, so a positional slice would grab the
    # wrong rows or miss them entirely.
    electron_dict = dict(zip(times, data["electron"].values))
    electron_high_dict = dict(zip(times, data["electron_high"].values))

    # Begin evaluating events
    maes = []
    o2p_lags = []
    o2t_lags = []
    ln10_lags = []
    pes = []
    for i, event in enumerate(event_times):

        # Get event times
        onset = event[0]
        bg_before_onset = times[times.index(onset) - 36]
        threshold = event[1]
        peak = event[2]

        # Get relevant portions of each time series for plotting
        times_o2p = times[times.index(bg_before_onset): times.index(peak) + 1]
        targets_o2p = slice_dictionary_keys(targets_test, bg_before_onset, peak)
        predictions_o2p = slice_dictionary_keys(predictions, bg_before_onset, peak)
        electron_o2p = slice_dictionary_keys(electron_dict, bg_before_onset, peak)
        electron_high_o2p = slice_dictionary_keys(electron_high_dict, bg_before_onset, peak)

        # Plot event
        fig, ax = plt.subplots()
        ax.set_ylabel('ln(Flux (/cc/s/sr))')
        ax.plot(list(targets_o2p.values()), '-b', label='Actual proton')
        ax.plot(list(predictions_o2p.values()), '-r', label='Predicted proton')
        ax.plot(list(electron_o2p.values()), '-m', label='Electron')
        ax.plot(list(electron_high_o2p.values()), '-y', label='High-energy electron')
        ax.plot([np.log(10)] * len(targets_o2p.values()), '--k')

        # Set x-axis to timestamps
        diff = int(ax.get_xticks()[1] - ax.get_xticks()[0])
        time_labels = [time[time.index('T') + 1: time.index('.')] for time in times_o2p[::diff]]
        if len(time_labels) == len(ax.get_xticks()) - 2:
            ax.set_xticks(ax.get_xticks()[1:-1])
        else:
            ax.set_xticks(ax.get_xticks()[1:-2])
        ax.set_xticklabels(time_labels, rotation='vertical')

        # Caption with date
        date_start = times_o2p[0][:times_o2p[0].index('T')]
        date_end = times_o2p[-1][:times_o2p[-1].index('T')]
        if date_start != date_end:
            ax.set_xlabel(f'Event from {date_start} to {date_end}')
        else:
            ax.set_xlabel(f'Event on {date_start}')

        fig.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
        fig.tight_layout()
        if path:
            plt.savefig(f'{path}/event{i + 1}.png')
        if display:
            plt.show()
        plt.close()

        # Now remove the 3 hours of background since they are not evaluated
        targets_o2p = slice_dictionary_keys(targets_test, onset, peak)
        predictions_o2p = slice_dictionary_keys(predictions, onset, peak)
        targets_o2t = slice_dictionary_keys(targets_test, onset, threshold)
        predictions_o2t = slice_dictionary_keys(predictions, onset, threshold)

        # Also convert from dictionary to numpy array; timestamps are no longer needed
        targets_o2p = np.array(list(targets_o2p.values()))
        predictions_o2p = np.array(list(predictions_o2p.values()))
        targets_o2t = np.array(list(targets_o2t.values()))
        predictions_o2t = np.array(list(predictions_o2t.values()))

        # Calculate stats
        maes.append(mae(targets_o2p, predictions_o2p, False))
        pes.append(pe(targets_o2p, predictions_o2p))
        o2p_lags.append(x_axis_error(targets_o2p, predictions_o2p, False))
        o2t_lags.append(x_axis_error(targets_o2t, predictions_o2t, False))
        ln10_lags.append(lag_ln10(targets_o2p, predictions_o2p))

    # Output metrics per event
    if path:
        outfile = open(f'{path}/results.txt', 'w')
        for i in range(len(event_times)):
            outfile.write(f"Event {i + 1}\nMAE = {maes[i]: 0.3f}\nO2P lag = {o2p_lags[i]: 0.3f}\n"
                          f"O2T lag = {o2t_lags[i]: 0.3f}\nln10 lag = {ln10_lags[i]: 0.3f}\n\n")
        outfile.write(f"Average MAE = {np.average(maes): 0.3f}\n")
        outfile.write(f"Average PE = {np.average(pes): 0.3f}\n")
        outfile.write(f"Average O2P lag = {np.average(o2p_lags): 0.3f}\n")
        outfile.write(f"Average O2T lag = {np.average(o2t_lags): 0.3f}\n")
        outfile.write(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}\n")
        outfile.close()

    # Output average metrics to standard output
    print(f"Average MAE = {np.average(maes): 0.3f}")
    print(f"Average PE = {np.average(pes): 0.3f}")
    print(f"Average O2P lag = {np.average(o2p_lags): 0.3f}")
    print(f"Average O2T lag = {np.average(o2t_lags): 0.3f}")
    print(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}")
    

# def main():

#     # Add arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', '--prediction_time', type=int, required=False,
#                         default=6, help="Number of timesteps ahead to predict")
#     parser.add_argument('-a', '--algorithm', type=str, required=False, default='regular',
#                         choices=['regular', 'rnn'], help="Algorithm to be used for intensity model (regular or RNN)")
#     parser.add_argument('-ph', '--phase_inputs', action='store_true', required=False,
#                         default=False, help="Whether or not to use phase inputs")
#     parser.add_argument('-lm', '--load_model', action='store_true', required=False,
#                         default=False, help="Whether or not to load trained model")
#     parser.add_argument('-lp', '--load_predictions', action='store_true', required=False,
#                         default=False, help="Whether or not to load predictions")
#     parser.add_argument('-p', '--path', type=str, required=False,
#                         default=None, help="Directory to load and store files")
#     parser.add_argument('-d', '--display', action='store_true', required=False, default=False,
#                         help="Whether or not to display event plots")
#     parser.add_argument('-s', '--seed', type=int, required=False, help='Seed for random number generator')

#     # Parse arguments
#     args = parser.parse_args()
#     prediction_time = args.prediction_time
#     algorithm = args.algorithm
#     use_phase_inputs = args.phase_inputs
#     load_models = args.load_model
#     load_predictions = args.load_predictions
#     path = args.path
#     display = args.display
#     seed = args.seed

#     # Pair inputs and outputs, and split into train/test
#     data = pd.read_csv('../Data/data.csv')
#     train, targets_train, test, targets_test = pair_input_output(data, use_phase_inputs, prediction_time)

#     # Reshape train/test depending on algorithm
#     if algorithm == 'regular':
#         train = np.array([instance.flatten() for instance in train])
#         test = np.array([instance.flatten() for instance in test])
#     else:
#         train = np.array([instance.T for instance in train])
#         test = np.array([instance.T for instance in test])

#     # Either load predictions, load model and get predictions, or train model and get predictions
#     if load_predictions:
#         if path:
#             predictions = load_series(f"{path}/predictions.txt")
#         else:
#             print("No path to load predictions from, exiting program.")
#             exit(0)
#     else:
#         if load_models:
#             if path:
#                 model = load_model(f"{path}/model")
#             else:
#                 print("No path to load model from, exiting program.")
#                 exit(0)
#         else:
#             model = train_model(train, targets_train, algorithm)
#             if path:
#                 model.save(f"{path}/model")

#         # Get and save predictions from model
#         predictions = model.predict(test)
#         predictions = predictions.flatten()
#         if path:
#             write_file(f"{path}/predictions.txt", predictions)

#     # Load events for evaluation
#     event_file = open('../Data/event_timestamps.txt', 'r')
#     lines = event_file.readlines()
#     event_times = [line.split() for line in lines]
#     event_file.close()

#     # Select only events which are in the test set
#     event_times_test = []
#     first_test_event_time = list(targets_test.keys())[0]
#     for event in event_times:
#         if event[0] >= first_test_event_time:
#             event_times_test.append(event)

#     # Evaluate predictions
#     target_times = list(targets_test.keys())
#     predictions = {target_times[i]: predictions[i] for i in range(len(predictions))}
#     evaluate(targets_test, predictions, event_times_test, data, path, display)


# if __name__ == "__main__":
#     main()
