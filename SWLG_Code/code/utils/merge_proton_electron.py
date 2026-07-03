import pandas as pd
import os
from utils.flux_integration import Integrating_Flux_l3i
from utils.flux_integration import Integrating_Flux
import numpy as np
L3I_COLUMNS = [
    'Year', 'Month', 'Day', 'DOY', 'Hour', 'Minute', 'status',
    'accum_time', 'P4', 'P8', 'P25', 'P41',
    'sys_p4', 'sys_p8', 'sys_p25', 'sys_p41',
    'stat_p4', 'stat_p8', 'stat_p25', 'stat_p41',
    'H4', 'H8', 'H25', 'H41',
    'sys_h4', 'sys_h8', 'sys_h25', 'sys_h41',
    'stat_h4', 'stat_h8', 'stat_h25', 'stat_h41'
]


def Merge_Proton_Electron(proton_path, e150_path, e300_path, output_path,
                          start_year=1997, end_year=2025):
    os.makedirs(output_path, exist_ok=True)

    print("Loading E150...")
    e150_df = pd.read_csv(e150_path)
    e150_df['Timestamp'] = pd.to_datetime(e150_df['Date Time']).dt.floor('5min')
    e150_df = e150_df.groupby('Timestamp')['E150'].mean().reset_index()

    print("Loading E300...")
    e300_df = pd.read_csv(e300_path)
    e300_df['Timestamp'] = pd.to_datetime(e300_df['Date Time']).dt.floor('5min')
    e300_df = e300_df.groupby('Timestamp')['E300'].mean().reset_index()

    for year in range(start_year, end_year + 1):
        filepath = os.path.join(proton_path, f"{year}.l3i")
        if not os.path.exists(filepath):
            print(f"Skipping {year} — file not found")
            continue

        print(f"Processing {year}...")

        proton = pd.read_csv(filepath, sep=r'\s+', comment='#',
                             header=None, names=L3I_COLUMNS)

        proton = proton[(proton["stat_p4"] != -999) & (proton["stat_p8"] != -999)]
        proton = proton[(proton["P4"] < 1e5) & (proton["P8"] < 1e5) &
                        (proton["P25"] < 1e5) & (proton["P41"] < 1e5)]
        proton = Integrating_Flux(proton)
        

        proton['Timestamp'] = pd.to_datetime(
            proton[['Year', 'Month', 'Day']].assign(
                hour=proton['Hour'], minute=proton['Minute']
            )
        )

        for col in ['P4', 'P8', 'P25', 'P41', 'P_tot']:
            proton.loc[proton[col] < 0, col] = None

        proton = proton[['Timestamp', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DOY',
                         'P4', 'P8', 'P25', 'P41', 'P_tot',
                         'stat_p4', 'stat_p8', 'stat_p25', 'stat_p41']]

        merged = pd.merge(proton, e150_df, on='Timestamp', how='left')

        if year <= 2017:
            merged = pd.merge(merged, e300_df, on='Timestamp', how='left')
        else:
            merged['E300'] = None

    #     for col in ["E150", "E300", "P_tot"]:
    #         merged[col] = np.log(
    #             np.clip(merged[col], 1e-6, None)
    # )
        out_file = os.path.join(output_path, f"merged_{year}.csv")
        merged.to_csv(out_file, index=False)
        print(f"  Saved {len(merged)} rows for {year}")

    print("Done.")