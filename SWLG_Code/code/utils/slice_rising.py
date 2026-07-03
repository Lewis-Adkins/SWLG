import pandas as pd

def Slice_Rising(a_data_df, a_phase_df):
    df = a_data_df.reset_index(drop=True).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    ts_index = pd.DatetimeIndex(df["Timestamp"])

    # use actual min/max, not first/last (data may be unsorted)
    data_start, data_end = ts_index.min(), ts_index.max()

<<<<<<< HEAD
=======
    print(f"SLICE: data range {data_start} to {data_end}", flush=True)
    print(f"SLICE: phase events = {len(a_phase_df)}, cols = {a_phase_df.columns.tolist()}", flush=True)
>>>>>>> b140e5d7f3685e2935e0eaca4085869434747d56

    rising_frames = []
    skipped_range = 0
    skipped_order = 0

    for event_counter, (_, storm) in enumerate(a_phase_df.iterrows()):
        onset_ts = pd.to_datetime(storm["Onset Timestamp"])
        peak_ts  = pd.to_datetime(storm["Peak Timestamp"])

        if not (data_start <= onset_ts <= data_end) or not (data_start <= peak_ts <= data_end):
            skipped_range += 1
            continue

        onset_pos = ts_index.get_indexer([onset_ts], method="nearest")[0]
        peak_pos  = ts_index.get_indexer([peak_ts],  method="nearest")[0]

        if peak_pos <= onset_pos:
            skipped_order += 1
            continue

        event_slice = df.iloc[onset_pos: peak_pos + 1].copy()
        event_slice["event_id"] = event_counter
        rising_frames.append(event_slice)

<<<<<<< HEAD

=======
    print(f"SLICE: matched {len(rising_frames)} events, "
          f"skipped {skipped_range} (out of range), {skipped_order} (bad order)", flush=True)
>>>>>>> b140e5d7f3685e2935e0eaca4085869434747d56

    if not rising_frames:
        return pd.DataFrame(columns=list(df.columns) + ["event_id"])

    return pd.concat(rising_frames, ignore_index=True)