"""
Shared rules for the data layout: reading the event list, the size of the test split, and which events land in it.
Used by the models (BaselineModel.test_events), SetUp and Analysis, so every stage follows the same rules.
"""


def read_event_timestamps(path: str = "data/event_timestamps.txt") -> list[list[str]]:
    # One [onset, threshold, peak, end] timestamp list per line (line 1 = event e1)
    with open(path, 'r') as f:
        return [line.split() for line in f.readlines()]


def n_test_samples(n_rows: int, prediction_time: int, train_split: float) -> int:
    # BaselineModel.pair_input_output makes one sample per t in range(24, n_rows - prediction_time) and
    # train_test_split keeps the last (1 - train_split) of them, so the test targets are the last n_test rows.
    n_samples = n_rows - prediction_time - 24
    return n_samples - int(train_split * n_samples)


def find_test_events(test_times, events: list[list[str]]) -> dict[int, tuple[int, int, int]]:
    # Events whose onset and peak are both in the test targets, found by timestamp (event_indices.txt only
    # fits data_0; the bootstrap datasets move the events). {event line index: (onset, threshold, peak) row
    # positions within the test targets}, in event_timestamps.txt order.
    position = {time: i for i, time in enumerate(test_times)}
    return {i: (position[onset], position.get(threshold), position[peak])
            for i, (onset, threshold, peak, _end) in enumerate(events)
            if onset in position and peak in position}
