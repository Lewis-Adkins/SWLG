from scipy.signal import find_peaks
import pandas as pd

def Timestamp_Proton_Flux(a_indices, a_df):

    
    #a_df = a_df.reset_index(drop=True)

    proton_fluxes = a_df["P_tot"]
    years         = a_df["Year"]
    DOYs          = a_df["DOY"]
    hours         = a_df["Hour"]
    fluxes    = proton_fluxes.iloc[a_indices]
    years     = years.loc[a_indices]
    DOY       = DOYs.loc[a_indices]
    hours     = hours.loc[a_indices]

    if "Minute"  in a_df.columns: 
        minutes         = a_df["Minute"]
        minutes         = minutes.loc[a_indices]
        timestamp = pd.to_datetime(years, format="%Y") + pd.to_timedelta(DOY - 1, unit="D") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
    else:
        timestamp = pd.to_datetime(years, format="%Y") + pd.to_timedelta(DOY - 1, unit="D") + pd.to_timedelta(hours, unit="h")
    return fluxes, timestamp


def Find_Proton_Flux_Phases(a_df, a_range, a_file_name):
    
    
    a_df = a_df.dropna()
    a_df = a_df.reset_index()
    
    proton_fluxes = a_df["P_tot"]
    years         = a_df["Year"]
    DOYs          = a_df["DOY"]
    hours         = a_df["Hour"]

    peak_indices, properties = find_peaks(proton_fluxes.to_list(), height=10, distance=a_range)

    peak_years     = years.iloc[peak_indices]
    peak_DOYs      = DOYs.iloc[peak_indices]
    peak_hours     = hours.iloc[peak_indices]
    peak_dates     = pd.to_datetime(peak_years, format="%Y") + pd.to_timedelta(peak_DOYs - 1, unit="D")

    if "Minute"  in a_df.columns: 
        minutes       = a_df["Minute"]
        peak_minutes = minutes.iloc[peak_indices]
        peak_timestamp = (peak_dates + pd.to_timedelta(peak_hours, unit="h") + pd.to_timedelta(peak_minutes, unit="m")).reset_index(drop=True)



    else: 
        peak_timestamp = (peak_dates + pd.to_timedelta(peak_hours, unit="h")).reset_index(drop=True)

    onset_indices     = []
    threshold_indices = []
    end_indices       = []

    for i in range(len(peak_indices)):
        peak_index = int(peak_indices[i])

        # --- Background: find quiet 24-hour window ---
        b = 0
        background_noise = proton_fluxes.iloc[peak_index - 24: peak_index]
        while (background_noise > 2).any():
            background_noise = proton_fluxes.iloc[peak_index - 24 - b: peak_index - b]
            b += 1

        mean_bg   = background_noise.mean()
        std_bg    = background_noise.std()
        onset_check = mean_bg + 3 * std_bg

        ###------------------------ Onset ------------------------###
        
        current_onset_index = peak_index
        while current_onset_index > 0 and proton_fluxes.iloc[current_onset_index] > onset_check:
            current_onset_index -= 1

        if current_onset_index <= peak_indices[i-1] and peak_indices[i-1] < peak_index :
    
                current_onset_index = proton_fluxes.iloc[int(peak_indices[i-1]): peak_index].idxmin()
                onset_indices.append(current_onset_index)
        else:

            onset_indices.append(current_onset_index)

        ###------------------------ Treshold ------------------------###

        current_threshold_index = current_onset_index
        while current_threshold_index < peak_index and proton_fluxes.iloc[current_threshold_index] < 10:

            current_threshold_index += 1

        threshold_indices.append(current_threshold_index)

        # --- End: walk forward from peak until flux < 2 ---
        current_end_index = peak_index
        while current_end_index < len(proton_fluxes) - 1 and proton_fluxes.iloc[current_end_index] > 2:
            current_end_index += 1
        end_indices.append(current_end_index)

    onset_fluxes,     onset_timestamp     = Timestamp_Proton_Flux(onset_indices, a_df)

    
    threshold_fluxes, threshold_timestamp = Timestamp_Proton_Flux(threshold_indices, a_df)
    end_fluxes,       end_timestamp       = Timestamp_Proton_Flux(end_indices, a_df)
    peak_fluxes = pd.Series(properties['peak_heights'])


    phases_df = pd.DataFrame({
        'Onset Timestamp':     onset_timestamp.reset_index(drop=True),
        'Onset Flux':          onset_fluxes.reset_index(drop=True),
        'Threshold Timestamp': threshold_timestamp.reset_index(drop=True),
        'Threshold Flux':      threshold_fluxes.reset_index(drop=True),
        'Peak Timestamp':      peak_timestamp,
        'Peak Flux':           peak_fluxes,
        'End Timestamp':       end_timestamp.reset_index(drop=True),
        'End Flux':            end_fluxes.reset_index(drop=True)
    })

    phases_df["Storm Duration"] = phases_df["End Timestamp"] - phases_df["Onset Timestamp"]
    phases_df = phases_df[phases_df["Onset Flux"] < 2]
    
    phases_df.to_csv("data/phases/" + a_file_name + "-" + str(a_range) + ".csv")

    return phases_df