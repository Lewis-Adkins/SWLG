
**Title:** Single Event Upset Investigations on the Flying Laptop Satellite Mission
**Authors:** Christopher Noeldeke, Maximulan Boettcher, Ulrich Mohr, Steffen Gaisser, Mikel Alvarez Rua, Jens Eickoff, Mike Leslie, MAtt Von Thun, Sabine Klinker, Renuganth Varatharajoo
**Publisher**: Science Direct
**DOI:** [10.1016/j.asr.2020.12.032](https://ui.adsabs.harvard.edu/link_gateway/2021AdSpR..67.2000N/doi:10.1016/j.asr.2020.12.032)
# Abstract

This paper presents the SEU that were observed in an inboard memory device of the LEO "Flying Laptop" satellite mission during its in-orbit operation. The SEU were mapped on the satellite orbital space with their root causes:
* trapped energetic protons leaking to low altitudes within South Atlantic Anomaly
* Galactic Cosmic Rays GCR
* Solar Energetic Particles

# 1. Introduction

The satellite was developed and operated by Institute of Space Systems at University of Stuttgart (Germany). It was launched on July 14th 2017 into a 600 km Sun Synchronous orbit. Focus of this paper is on the Static Random Access Memory (SRAM).

# 2. Observation Data and Data Processing

During the routine telemetry acquisition SRAM SEU were observed and corrected by the Error Detection and Correction (EDAC) system, a 7-bit check sum for each 32 bit word of data both stored in the SRAM. EDAC corrects single bit flips and to detect double bit flips resulting from SEUs

Linear Energy Transfer of protons on silicon is low therefore proton induced upsets are not caused by direct ionization but by inelastic scattering and nuclear reactions with the lattice atoms of the tracer (semiconductor material).

# 3. The Radiation Environment

Over 286 days, 2128 SEUs were catalogued, 68% within SAA

## 3.1 Solar Energetic Particles (SEPs)

Major constituents of SEPs are protons and $\alpha$ particles with energies several 100 MeV per nucleon. Flux of SEP is intermittent with durations of hours or Days. highest probable Solar Particle Event occurring  is in the magnetic polar regions where the vertical cut off rigidity is lowest. Because SEP are extraterrestrial SEP penetration to a satellite on LEO is strongly influenced by the satellite momentary position within the magnetosphere as well as by the incidence direction and the SEPs rigidity.

During Observations SEPs with considerable geoeffective proton fluxes above 4.2 MeV were present. Origins is active region on Sun. Data from NOAA NGDC (2017) *add link*

No precise time tags available for the SUE observed on the satellite for September 2017 there is no direct evidence where the SEP might have hit the satellite

##  3.2 Galactic Cosmic Rays (GCRs)

GCRs consist of completely ionized atoms of virtually all species from hydrogen to uranium. Isotropic outside earths magnetosphere. GCR heavier than hydrogen causes direct ionization in electronic devices. highest probably hit around LEO sats around magnetic polar regions of earth similar to SEP.

## 3.3 Vertical cut- off rigidity

abundancies predicted by the CREME96 model

# 4. SEU Rate Analysis

To analyze SEU rates as a function of the satellite position, standard models describing the flux spectra of trapped protons and GCRs were used. Trapped protons were analyzed by AP9 model while GCR both used CREME96 and ISO 15390 models and integrated into the SPENVIS software.

To model Earths magnetic field the International Geomagnetic Reference Field (IGRF) model was used with corresponding observation period.

A 1 Mbit  Complementary Metal - Oxide Semiconductor (CMOS) SRAM test data was used. A Weibull fit paramter with shape parameter $S = 2.5$ threshold energy $E_0 = 5$ MeV, width $W = 20$ MeV and limit cross section $\sigma_{lim} = 3.3 \cdot 10^{-14}$ cm^2/bit

Proton induced upset rates were assessed using Weibull fitting o proton cross sections $\sigma(E)$ as a function of particle energy, and GCRs induced upsets rates used Weibull firring of cross section as a function of $\sigma(LET)$ of the Linear Energy Transfer (LET). 

Sectorial analysis model for satellite unavailable, sheilding was assesed  using SHIELDOSE2 algorithm

Heavy ion cross sections available from test data of the UT8ER512K32 chip carries same die as the UT8R2M39 SRAM on satellite.

assessing SEU rate associated to the SEP events in September 201, data from GOES-15 EPEAD instrument were evaluated. 

# Data Statement







