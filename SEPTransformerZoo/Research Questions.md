**Overarching question:** Can ML forecasts of >10 MeV protons give satellite operators enough warning to safe-mode before single-event upsets occur?
---
[[LaNeve 2024]]
* *Satellite operators are faced with a plethora of information, limited time for decision making, and limited knowledge of historical space weather relative to their other areas of expertise, causing the need for confident forecasts* (p6–7)
* Only 10% of satellite operators placed satellites into safe mode after a forecast; 58% increased staff situational awareness (p19).
* One operator's only "kinetic responses" in their history were *repositioning the satellite to avoid reaching proton thresholds* (p19).
* One operator avoided safe mode *citing that the loss of services would outweigh the risk of damage, and that the customers almost always opted for continued service* (p19).
* *the tyranny of distance between operators and equipment* (p24)
* Operators *identified the need for more accurate, localized, space weather warnings* (p2)


# Why SEP Forecasting?

[[Schrijver 2015]]
* *solar energetic-particle (SEP) storms, for example, impact satellites directly, while expansion of the terrestrial upper atmosphere by magnetospheric variability (through Joule heating) may affect low-orbiting satellites by modifying their orbits through increased drag which is an issue for on-orbit operations, collision avoidance, and could eventually lead to early re-entry*
* *A series of such storms during the 2003 October-November time frame, for example, saw considerable impacts on satellites through electronic single event upsets (SEUs), solar-array degradation, modified orbit dynamics for spacecraft in low-Earth orbits, and noise on both housekeeping data and instrument data*
* *the possibility to disrupt HF communications that are used by civil aviation in remote regions, especially over the oceans and poles.*
* *SEPs is very significant as their impact is greatest in the polar regions (where satcom backup is not currently available) and is long-lived (up to days) so that procedural measures are insufficient. Thus polar flights are diverted to other less vulnerable routes during SEP events, imposing significant extra costs on airlines (around $100k per diversion; National Research Council, 2008).*
* *SEP where these have strong fluxes at energies >500MeV) that penetrate to the Earth's atmosphere and interact with atmospheric neutral particles. Such collisions produce cascades of neutrons and ions that can interact with aircraft to produce single event effects (SEE) and increased dose rate for passengers and crew*
* *On a par with establishing the structure of the erupting solar field as it begins to propagate into the heliosphere is establishing how incoming solar-wind field structures interact with the magnetospheric field to drive geomagnetic variability either immediately or with delay after storing energy in the magnetotail. L1 in-situ particle and field measurements are critical in providing input to models for near-term forecasts, and for validation of longer-term forecasts of storms advancing through the heliosphere. Therefore, the continuity of L1 measurements should be ensured so that besides the needs of operational services also the needs from the research community are taken into account.*
* *energies below about 1GeV penetrate only poleward of about 55 degrees magnetic latitude* (§4.4, p19). This is why >10 MeV protons threaten polar and high-inclination LEO satellites, while equatorial LEO is largely shielded.
* *The prompt SEPs represent the most difficult forecasting problem* (Appendix D, Goal SH-4, p42)
* Goal SH-4b, Predict Rapidly Arriving SEPs (time scale minutes to hours): *relativistic and mildly relativistic particles (neutron monitors and GOES) to establish connectivity and forewarning of future, less energetic particles* (p43). This is the electron-precursor approach used here.

[[Tobiska 2015]]
* *The NOAA Space Weather Prediction Center issues a Watch (long-lead-time geomagnetic activity prediction), a Warning (some condition is expected), or an Alert (event threshold is crossed). A Watch is provided only for geomagnetic storms and not SEP events.*

[[Kasapis 2026]]
* *Space weather disturbances, particularly their more hazardous aspects, are driven by highly energetic solar events such as flares, coronal mass ejections (CMEs), and solar energetic particle (SEP) events*
* *SEPs are central to space weather concerns as they represent a major radiation hazard for astronauts, particularly during extravehicular activities, and for passengers and crew on high-latitude flights, Damage satellites, disrupt comms and impair navigation. Moon or mars mission no shielding from atmosphere*
* *SEP events pose significant challenges to space weather forecasting due to their rapid onset and variability and potential impact on both technological systems and human health*

[[Lohmeyer 2013]]
* *Particles greater than 10 MeV are capable of penetrating surface of a satellite. main sources are solar energetic protons (SPE) and galactic cosmic rays (GCRs)*
* *High energy protons were capable of causing SEUs*

[[Wilkinson 1991]]
* *Three main sources of particle radiation in space environment*
	1. *Galactic Cosmic Rays (GCRs)*
	2. ==*Energetic Solar Particles from solar flares*==
	3. *Energetic particles from radiation belts from Earths magnetosphere*
* *Primary cause of TDRS-1 SEUs during solar flares (2)*

[[Eastwood 2017]]
* *September 1859The "Carrington" event is the benchmark for extreme space weather studies. The solar flare, the geomagnetic storm, and the ==energetic particle flux== associated with this event make it one of the largest on record. Note that many crucial parameters were not measured directly, so its precise properties are subject to uncertainty. In particular, estimating the strength of the geomagnetic storm associated with the Carrington event has attracted some debate; initial estimates should be disregarded in favor of more recent analysis*
* *An extreme solar flare and coronal mass ejection caused very significant radio blackouts, ==solar radiation storms==, and a major geomagnetic storm.*
* *disruption to communications, specifically to the military, and marked the start of a significant U.S. investment in space weather monitoring*
* *SEPs can cause displacement damage (reducing device performance) and single event effects (SEEs), which are a growing issue as devices are miniaturized.*
* *During the 2003 Halloween storms, 47 satellites reported anomalies (out of 450 in orbit, i.e., ∼10%), one scientific satellite was lost, and 10 satellites lost operational service for more than one day. Complete losses have thus been rare since satellites are designed to tolerate a total dose over some lifetime, with good safety margins: temporary outages and fleet aging are both more likely.*

[[Gopalswamy 2024]]
* *SEP are major radiation hazards in space both inside earths magnetosphere and outside the up space. risk for HF comms, satellite operations and biological systems.*
* *SEP events result in frequent spacecraft anomalies. The frequency of spacecraft anomalies depends on the intensity energy range of SEP events, spacecraft altitude and spacecraft inclination*



# Why Machine Learning?

[[Whitman 2022]]
* Physics based models are computationally intensive, and the ability to produce a prediction ''in real time'' may be limited by the resources available, so they may not be readily incorporated into a forecasting workflow.

[[Kasapis 2026]]
* physics based models simulate particle acceleration and transport processes but require substantial computational resources
* empirical and statistical models provide predictions but rely heavily on historical correlations
* ML emerged as a powerful alternative tool uncovering non linear patterns across heliospheric datasets

[[Yu 2026]]
* *Physics and empirical models are limited by:*
	1. *uncertain inputs*
	2. *incomplete observational coverage*
	3. *tradeoff between realism and forecast timeliness*
* *ML Combines diverse solar heliospheric observations under a unified predictive framework*
	* *Across the tested configurations, the strongest performance was obtained when magnetic-field information was combined with historical proton flux and radiative indicators of eruptive activity*

[[Camporeale 2019]]
"*We have sophisticated [[First Principles]] physics based models to understand Space Weather events but it comes with issues*
1. *separation of scales in space and time involved.*
2. *short time lag between storm and satellite response*
3. *computational cost "*

[[Schrijver 2025]]
* *Statistical ionosphere models tuned with data from ground-based networks or low Earth orbiting (LEO) satellites can in many cases provide relatively good results for nowcasts or short-term forecasts… However, with increasing lead times more comprehensive physics-based modeling with data assimilation is required.* (§4.3, p18). Data-driven models are strongest at short lead times.

# Motivation

Near-relativistic electrons (SOHO/EPHIN >0.25 and >0.67 MeV) reach L1 within roughly 10–13 min, while >10 MeV protons take roughly 30–65 min (scatter-free, ~1.15 AU field-line path; ~30 min for the fastest protons in the integral, ~65 min at 10 MeV). That is a ~20–55 min head start, which is the warning window a forecaster can exploit. Each RQ below asks how much of it a model turns into usable warning.

---

# RQ1: Which approach forecasts most accurately?

Do deep-learning models (NN, RNN, RoPE, Sin, Zero Transformers, AutoGluon TFT, Chronos-2) forecast >10 MeV proton flux 30 min (t+6) and 60 min (t+12) ahead more accurately than statistical baselines (linear regression, persistence)?

## RQ1.1 Accuracy and timing (Welch t-test, assuming no equivalent variance across model population metrics)

1. $H_0 \text{ F1}: \mu_c \leq \mu_b$
2. $H_0 \text{ PE}: \mu_c \leq \mu_b$
3. $H_0 \text{ Lag}: |\mu_c| \geq |\mu_b|$
4. $H_0 \text{ MAE}: \mu_c \leq \mu_b$
5. $H_0 \text{ HSS}_1 : \mu_c \leq \mu_b$
6. $H_0 \text{ HSS}_2 : \mu_c \leq \mu_b$
7. $H_0 \text{ TSS}_2 : \mu_c \leq \mu_b$
8. $H_0  \text{ } r^2 : \mu_c \leq \mu_b$
9. $H_0  \text{ MSE}  : \mu_c \leq \mu_b$
10. $H_0  \text{ RMSE}  : \mu_c \leq \mu_b$

- Sample unit: one event-averaged score per bootstrap dataset (seeds averaged).

## RQ1.2 Consistency (Levene's test)

- $H_0: \sigma_\text{champ}^2 \geq \sigma_\text{base}^2$

## RQ1 Figures

**Preliminary: all models are 1-epoch runs and are not accucrate to comparing with torres and persistence**
---
**Scorecard.** Mean score of each model on each metric over the 10 bootstrap datasets, at 30 and 60 minutes ahead. Cell colour shows the effect size against persistence (Cohen's *d*; blue = better, orange = worse).

![RQ1 heatmap](results/metrics_analysis/rq1/rq1_heatmap_nophases.png)

**Spread across datasets.** Each dot is one bootstrap dataset's score, with box plots showing the median and quartiles for each model and horizon. The vertical lines mark persistence's median, so the plot shows how consistently each model beats or trails it.

![RQ1 box plots](results/metrics_analysis/rq1/rq1_box_nophases.png)

**Significance.** Each point is one model on one metric: x is the effect size against persistence, and y is the Bonferroni-adjusted p-value (two-sided Welch t-test) on a −log₁₀ scale. Points in the upper right are significantly and substantially better than persistence. The numbers behind the plot are in `rq1_tests_nophases.csv`.

![RQ1 volcano](results/metrics_analysis/rq1/rq1_volcano_nophases.png)

---

# RQ2: Is the warning fast enough for operators to safe-mode a satellite?

What is the end-to-end response time from SEP detection at L1 to satellite safe mode, and do ML forecasts of >10 MeV protons provide warning that exceeds it for GEO and polar LEO satellites? Builds on [[LaNeve 2024]], which recorded *what* operators did during the 2024 solar maximum but not *how long* each step takes.

## RQ2.1 The response pipeline


**Condition for usable warning:**

$$T_\text{warn} \geq R = T_\text{data} + T_\text{alert} + T_\text{decide} + T_\text{uplink} + T_\text{safe}$$

- **Data source for $T_\text{decide}$, $T_\text{uplink}$, $T_\text{safe}$:** extend the [[LaNeve 2024]] survey (chrono-storm.com) with timing questions for satellite operators.

## RQ2.2 Do champion models clear the threshold more often?

- **Warning time** = observed threshold-crossing time − issue time of the first forecast predicting the crossing.
- **Usable warning** = warning time ≥ $R$ (missed events count as 0), evaluated separately for the GEO and polar LEO values of $R$.
- Tests:
  - $H_0: \mu_{\text{warn},c} \leq \mu_{\text{warn},b}$ (Welch)
  - $H_0: p_c \leq p_b$ for the share of events with usable warning (McNemar, paired on the same events)
- Persistence gives about zero warning by construction, so linear regression is the main reference baseline here.
- **Context:** compare against NOAA SWPC proton alerts (issued at or after threshold crossing) and Posner's REleASE (up to about 1 h of warning).

## RQ2 Figure

*Preliminary:  all other models are 1-epoch runs.*
---
[[LaNeve 2024]]
* *Satellite operators are faced with a plethora of information, limited time for decision making, and limited knowledge of historical space weather relative to their other areas of expertise, causing the need for confident forecasts* (p6–7)
* Only 10% of satellite operators placed satellites into safe mode after a forecast; 58% increased staff situational awareness (p19).
* One operator's only "kinetic responses" in their history were *repositioning the satellite to avoid reaching proton thresholds* (p19).
* One operator avoided safe mode *citing that the loss of services would outweigh the risk of damage, and that the customers almost always opted for continued service* (p19).
* *the tyranny of distance between operators and equipment* (p24)
* Operators *identified the need for more accurate, localized, space weather warnings* (p2)
**Usable warning against response time.** For each assumed satellite response time $R$, the share of SEP events where an alert (a forecast reaching 10 pfu) was issued at least $R$ minutes before the observed crossing. Lines are means over the bootstrap datasets. The grey band marks the electron head start (~17–55 min).

![RQ2 warning vs response time](results/metrics_analysis/rq2/rq2_warning_nophases.png)

---

# RQ3: Sensitivity to forecast horizon. How far ahead can we forecast?

How does skill decay from t+6 → t+12 → t+18 → t+24 (30 min to 2 h)? At what horizon does each model stop beating persistence?

- **Prediction:** if models exploit the electron head start, skill should collapse once the horizon exceeds the electron–proton arrival gap (~20–55 min). Note that t+12 (60 min) already sits at or beyond this gap.
- **Output:** each model's *maximum useful lead time*, the largest horizon where it still beats persistence significantly.
- **Analysis:** metric vs horizon curves with bootstrap 95% CIs, plus the RQ1.1 tests and RQ2 warning-time tests repeated at each horizon.

---

# RQ4: Method design. How much history does a model need?

Does a longer input window (6, 12, 25, 48 and 96 steps, i.e. 30 min to 8 h) improve forecasts? Do attention-based models use long context better than NN, RNN and linear regression?

- **Prediction:** gains level off once the window covers the electron onset. If Transformers don't benefit from longer context, that is a finding in itself.
- **Analysis:** within each model, Friedman test across window sizes (paired on bootstrap datasets), then pairwise Wilcoxon (Holm-corrected). Report the window where each model levels off.

---

# RQ5: Model scaling. Do the models improve with more capacity and tuning?

How sensitive is forecast skill to model size and training hyperparameters?

| Hyperparameter | Current value |
|---|---|
| `dim_val` | 16 |
| `n_encoder_layers` | 1 |
| `n_heads` | 2 |
| `learning_rate` | 0.001 |

- **Design:** vary one hyperparameter at a time around the current configuration. Record parameter count and train time alongside each metric.
- **Analysis:** metric vs parameter count curves with bootstrap 95% CIs. Friedman test across settings within each model, then pairwise Wilcoxon (Holm-corrected).
- **Question for the reader:** does a bigger Transformer beat the baselines by more, or is the dataset too small for extra capacity to help?

---

## Scope notes

- **Data:** SOHO/COSTEP-EPHIN at L1, 5-min cadence, 15-min exponential average (Torres et al.). Inputs are the >0.25 MeV and >0.67 MeV electron intensities plus their derivatives. Target is the >10 MeV integral proton flux, built from the P8 (7.8–25 MeV), P25 (25–40.9 MeV) and P41 (40.9–53 MeV) channels minus the 7.8–10 MeV part (power-law interpolation from P4 and P8).
- **Events:** 39 SEP events with peak >10 pfu, December 1995 – January 2002 (Torres Table 1). Threshold is 10 pfu on the >10 MeV integral flux, the same definition as NOAA's S1 event threshold.
- **Fixed settings:** `batch_size` is held at the Torres baseline value (32) and is not part of the RQ5 sweep.
- **Compute:** if needed, run RQ3, RQ4 and RQ5 on a subset (linear regression, NN, RNN, best Transformer from RQ1).

## Future work (out of scope)

### FW1: Do additional characteristics of a solar storm, and observations off the Sun–Earth line, improve forecasts?

Does adding (a) remote-sensing precursors (flare X-rays, CME speed and width, type II/III radio bursts), (b) in-situ L1 solar wind (plasma, magnetic field), and (c) energetic-particle observations from spacecraft **off the Sun–Earth line** (STEREO-A, Solar Orbiter, Parker Solar Probe) to the EPHIN electron inputs improve the accuracy (RQ1) and warning time (RQ2) of >10 MeV proton forecasts?

[[Gopalswamy 2024]] [[Type III Radio Bursts]] [[Corona Mass Ejections]] [[Schrijver 2025]]

- **Gap:** SEP ML models so far use only near-Earth inputs (L1, GEO, SDO), e.g. SEPNET [[Yu 2026]] and MEMPSEP. None use particle data from spacecraft off the Sun–Earth line, which the COSPAR/ILWS roadmap calls for (Goal SH-4c) [[Schrijver 2025]].
- **Why it matters:** remote-sensing signals (X-rays, radio bursts, CMEs) arrive before the protons and could extend warning time. Off-line spacecraft can be magnetically connected to a source region before Earth is, which could help with poorly connected eastern events.
- **Approach:** add one feature group at a time to the RQ1 models and test the change in accuracy and warning time.
- **Limitation:** STEREO (2006) and Solar Orbiter (2020) don't overlap the 1995–2002 Torres events, so this needs a newer event list.

### FW2: Can real-time ML forecasts from current L1 spacecraft give enough warning?

Can models trained on historical SOHO/EPHIN data, run in real time on operational L1 data (SOLAR-1, with ACE and DSCOVR as backups), give at least $R$ minutes of warning (RQ2) for satellites in LEO/GEO?

- **Why SOLAR-1:** SOLAR-1 (launched September 2025 as SWFO-L1) is NOAA's operational replacement for ACE and DSCOVR at L1. It addresses two needs from the COSPAR/ILWS roadmap [[Schrijver 2025]]:
  1. **Continuity of L1 measurements.** *"L1 in-situ particle and field measurements are critical in providing input to models for near-term forecasts… Therefore, the continuity of L1 measurements should be ensured"* (§4.2.1, p16–17). SOHO (1995) and ACE (1997) are ageing research missions; SOLAR-1 is a dedicated operational successor.
  2. **Near-real-time data.** The roadmap emphasises *"the importance of near-real-time data availability"* (§5.1, p23) as a general recommendation for space weather data. SOLAR-1 is built for operational real-time delivery, which is what a live warning system needs (shrinking $T_\text{data}$ in RQ2).
- **Main obstacle:** SOLAR-1/STIS measures lower energies than EPHIN (electrons 25–250 keV, ions 25 keV–6 MeV, so no >10 MeV protons). Inputs must be cross-calibrated or the models retrained, with the >10 MeV target taken from GOES-19/SEISS.
- **Starting point:** Chronos-2 runs without training, so it can be tested on new instruments first.
- **Link to RQ2:** real-time operation shrinks $T_\text{data}$ and $T_\text{alert}$; an automated on-board or pre-authorised response would remove $T_\text{decide}$ and $T_\text{uplink}$ for polar LEO.

### FW3: Can proton flux forecasts be turned into SEU probabilities?

Can forecast >10 MeV proton flux be converted into the probability of a single-event upset on a satellite in the next 30–60 minutes, by orbit?
- **Gap:** very few studies link SEP forecasts to actual SEUs, because satellite anomaly data is proprietary. *"Unfortunately, detailed information on satellite anomalies is generally not released by companies that design or operate satellites"* [[Schrijver 2025]] (§5.2, p25). Operators *"may hold certain information as controlled information, thus limiting the responses"* [[LaNeve 2024]] (p25). The Inmarsat record is one of the few public exceptions.
- **Need for telemetry data:** linking SEPs to SEUs needs time-stamped satellite telemetry, not just anomaly reports:
  - SEU indicators: memory error-correction (EDAC) counts, bit-flip/scrub logs, processor resets, safe-mode entries.
  - Context: orbit position (to know when the satellite was outside geomagnetic shielding), shielding and device type.
  - Roadmap support: *"there is a big gap between the knowledge of the space environment and the statistics of satellite anomalies"* (§5.2, p24) and *"the correlation is often based on indirect evidence"* (§5.2, p25) [[Schrijver 2025]].
  - Public telemetry lacks timing: NASA's open-source telemanom dataset (SMAP and MSL; 82 channels, 105 labelled anomalies, including radiation channels) states *"All data has been anonymized with regard to time and all telemetry values are pre-scaled between (-1,1)"*, so anomalies cannot be matched to SEP events (Hundman et al., 2018; [NASA NPO-50838-1](https://software.nasa.gov/software/NPO-50838-1)).
  - Other telemetry is withheld: Yu et al. (2025, *Meas. Sci. Technol.* 36 015128) tested spacecraft anomaly detection on two telemetry datasets that *"cannot be made publicly available upon publication because they contain commercially sensitive information"* ([doi:10.1088/1361-6501/ad9495](https://iopscience.iop.org/article/10.1088/1361-6501/ad9495)).
  - Methods are ready: ML telemetry anomaly detectors (telemanom's LSTM forecast-error method; Yu et al.'s MINE + LSTM-VAE with attention) could flag SEU-like anomalies once time-stamped telemetry is obtained. Neither study looks at radiation or space-weather causes.
- **Evidence that SEPs drive SEUs:** [[Lohmeyer 2013]] used Inmarsat's archive of 226+ SEUs over about 20 years to test how often upsets fall within set time windows after >10 MeV solar proton events, and found high-energy protons capable of causing SEUs.
- **Approach:** SEU rate $\lambda(t) = \int \sigma(E)\,\phi(E,t)\,dE$ from the device cross-section $\sigma(E)$ and the forecast proton flux $\phi$, then a Poisson probability $P(\geq 1\ \text{SEU}) = 1 - e^{-\int \lambda\,dt}$ over the warning window.
- **Why it matters:** operators could safe-mode on "P(SEU) above a set threshold" rather than "flux above 10 pfu", giving the specific, satellite-level warning operators asked for in [[LaNeve 2024]].
- **Validation:** compare forecast SEU probabilities against the Lohmeyer Inmarsat SEU record.
- **Limitations:** SEU cross-sections depend mostly on higher energies than >10 MeV, so the proton spectrum is needed (e.g. EPHIN P25/P41 as extra targets). Cross-sections vary by device. Validation is limited to whatever SEU data can be obtained (Inmarsat, or operators willing to share under agreement).

### Other future work

- d_proton (proton derivative) as an extra input.
- A different solar cycle, for generalisation.
- Coupling with or comparing against CCMC physics models (EPREM, iPATH, HESPERIA REleASE).
