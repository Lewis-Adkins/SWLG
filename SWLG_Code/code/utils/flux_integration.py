'''
# Power Law Spectrum
**[Torres 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024SW003921)**: The 
10 MeV proton intensity is calculated from the differential fluxes in four energy (P4, P8, P25, and P41) channels through an integration over the energy. The P8 proton channel covers an energy range of 7.8–25 MeV, P25 covers 25–40.9 MeV, P41 covers 40.9–53 MeV, while the P4 channel covers 4.3–7.8 MeV. We assume that particles above the P41 channel coverage make little contribution to the integral flux of MeV protons. The integral flux is calculated by summing energy-integrated fluxes in the last three (P8, P25, and P41) channels minus the amount between 7.8 and 10 MeV, which is determined by the integration of a power-law spectrum interpolated using the differential fluxes in the first two channels (P4 and P8) and their mean energies. We use the first two channels to do the interpolation because their fluxes have relatively low error bars due to high count rates in low-energy channels.

Too find channel between 7.8- 10 MeV we fall the Power Law Scale and make use of its invariance:

$\Phi_n = \alpha \nu_n^\beta$ (1)

Using the channels P4 and P8:

$\Phi_4 = \alpha \nu_4^\beta$ (2) 

$\Phi_8 = \alpha \nu_8^\beta$ (3)

Converting frequency to energy: $\nu_n = \frac{\bar{E}_n}{h}$ (4), $h = 4.136 \times 10^{-21} \text{Mev/s}$

Rewriting (2): $\alpha = \frac{\Phi_4}{\nu_4^\beta}$

And then plugging into (3): $\Phi_8 = \Phi_4 \left(\frac{\nu_8}{\nu_4} \right)^\beta$ (5)

Solving for $\beta$: $\beta = \frac{\log [\Phi_8 / \Phi_4]}{\log [\nu_8 / \nu_4]}$

Solving for $\alpha$: $\alpha = \frac{\Phi_4}{\nu_4^\beta}$ With newly discovered $\beta$

Now we can solve the flux over any range we want using our new parameters $\alpha, \beta$: $\Phi_x = \alpha \nu_x^\beta$
'''

import numpy as np
import pandas as pd

def Power_Law_Spectrum_Flux(a_E1, a_E2, a_p4_flux, a_p4_avg_E, a_p8_flux, a_p8_avg_E):
    """Integrate power law spectrum from E1 to E2 MeV"""
    beta = np.log(a_p8_flux / a_p4_flux) / np.log(a_p8_avg_E / a_p4_avg_E)
    alpha = a_p4_flux / (a_p4_avg_E ** beta)
    
    if beta == -1:
        integral = alpha * np.log(a_E2 / a_E1)
    else:
        integral = alpha * (a_E2**(beta+1) - a_E1**(beta+1)) / (beta + 1)
    
    return integral

def Integrating_Flux(a_df):
    p4_avg_energy = (4.3 + 7.8) / 2   # MeV
    p8_avg_energy = (7.8 + 25) / 2     # MeV

    # Channel energy widths
    p8_dE  = 25.0 - 7.8    # 17.2 MeV
    p25_dE = 40.9 - 25.0    # 15.9 MeV
    p41_dE = 53.0 - 40.9    # 12.1 MeV

    # Integrate each channel: differential flux × energy width
    a_df["P8_int"]  = a_df["P8"]  * p8_dE
    a_df["P25_int"] = a_df["P25"] * p25_dE
    a_df["P41_int"] = a_df["P41"] * p41_dE

    # Subtract 7.8-10 MeV portion using power law interpolation
    a_df["Px"] = a_df.apply(
        lambda row: Power_Law_Spectrum_Flux(
            7.8, 10.0,
            row["P4"], p4_avg_energy,
            row["P8"], p8_avg_energy
        ),
        axis=1
    )

    # Total >10 MeV integral flux
    a_df["P_tot"] = a_df["P8_int"] + a_df["P25_int"] + a_df["P41_int"] - a_df["Px"]


    if "Minute" in a_df.columns:
        
        a_df["Timestamp"] = pd.to_datetime(a_df["Year"], format="%Y") + \
                            pd.to_timedelta(a_df["DOY"] - 1, unit="D") + \
                            pd.to_timedelta(a_df["Hour"], unit="h") +\
                            pd.to_timedelta(a_df["Minute"], unit = "m")
    else:
        a_df["Timestamp"] = pd.to_datetime(a_df["Year"], format="%Y") + \
                            pd.to_timedelta(a_df["DOY"] - 1, unit="D") + \
                            pd.to_timedelta(a_df["Hour"], unit="h")        

    # Remove error values
# Remove rows where any channel has fill values (original fill = 1e6 range)
    a_df = a_df[(a_df["P4"] < 1e5) & (a_df["P8"] < 1e5) & (a_df["P25"] < 1e5) & (a_df["P41"] < 1e5)]

    # Remove negative P_tot values
    a_df = a_df[a_df["P_tot"] >= 0]

    return a_df

def Integrating_Flux_l3i(a_df):
    p4_avg_energy = (4.3 + 7.8) / 2   # MeV
    p8_avg_energy = (7.8 + 25) / 2     # MeV


    # Subtract 7.8-10 MeV portion using power law interpolation
    a_df["px"] = a_df.apply(
        lambda row: Power_Law_Spectrum_Flux(
            7.8, 10.0,
            row["int_p4"], p4_avg_energy,
            row["int_p8"], p8_avg_energy
        ),
        axis=1
    )

    # Total >10 MeV integral flux
    a_df["p_tot"] = a_df["int_p8"] + a_df["int_p25"] + a_df["int_p41"] - a_df["px"]

    a_df["Timestamp"] = pd.to_datetime(a_df["year"], format="%Y") + \
                        pd.to_timedelta(a_df["doy"] - 1, unit="D") + \
                        pd.to_timedelta(a_df["hour"], unit="h") +\
                        pd.to_timedelta(a_df["minute"], unit = "m")

    # Remove error values
# Remove rows where any channel has fill values (original fill = 1e6 range)
    a_df = a_df[(a_df["int_p4"] < 1e5) & (a_df["int_p8"] < 1e5) & (a_df["int_p25"] < 1e5) & (a_df["int_p41"] < 1e5)]

    # Remove negative P_tot values
    a_df = a_df[a_df["p_tot"] >= 0]

    return a_df    
