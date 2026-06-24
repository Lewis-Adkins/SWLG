import numpy as np
import pandas as pd

def Slice_Rising(a_data_df, a_phase_df):
    rising_indices = np.array([[], []], dtype=np.int32)

    for i, storm in a_phase_df.iterrows():
        if (storm["Onset Timestamp"] in a_data_df["Timestamp"].values and
            storm["Peak Timestamp"] in a_data_df["Timestamp"].values):
            onset_idx = a_data_df["Timestamp"].index[a_data_df["Timestamp"] == storm["Onset Timestamp"]].tolist()[0]
            peak_idx  = a_data_df["Timestamp"].index[a_data_df["Timestamp"] == storm["Peak Timestamp"]].tolist()[0]
            rising_index = np.array([[onset_idx], [peak_idx]], dtype=np.int32)
            rising_indices = np.append(rising_indices, rising_index, axis=1)

    rising_flux_df = pd.DataFrame([])
    for i in range(rising_indices.shape[1]):
        start, end = rising_indices[0][i], rising_indices[1][i]
        event_slice = a_data_df.iloc[start:end].copy()
        event_slice["event_id"] = i          # <-- this is the line that fixes the error
        rising_flux_df = pd.concat([rising_flux_df, event_slice], ignore_index=True)

    return rising_flux_df