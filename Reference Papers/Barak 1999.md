**Title:** On the Figure of Merit Model for SEU Rate Calculations
**Authors:** J. Barak, R.A. Reed, and K.A. LaBel
**Publisher:** IEEE TRANSACTIONS ON NUCLEAR SCIENCE
**DOI:** [10.1109/23.819114](https://doi.org/10.1109/23.819114)


# Abstract

A one parameter characterization  of a device by the Figure of Merit (FOM), a numerical value used to evaluate, compare, and optimize the performance, efficiency, or quality of systems, materials, or devices. This parameter was sufficient to estimate the SEU rate in almost all orbits. We study the FOM concept and compare FOM model with other models. The FOM parameter gives a good agreement of SEU rates cross section plots of devices. High portion of proton flux coming from low energy protons and very SEU - hard devices causes poor results.

# I. Introduction

Measuring SEU rates $R$ of a device is done by measuring cross section for heavy ions and protons and folding them with ion fluxes in given orbit. We will be describing the Petersen method due to simplicity and the large number of cases which validated it.

Peterson demonstrates how the FOM can serve as a single number to estimate SEU rates. its unexpected due to:
1.  devices having different critical charge, geometry  etc.
2. orbits have different spectra of LET and proton energies
3. different shielding changes the spectra

Peterson used generic device characteristics and comparted the FOM prediction with integral rectangular parallelepiped results (IRPP). he also comparted the model for actrual devices with CREME96 with Weibull parameters

In this analysis we use CREME96 (solar minum and solar quiet) for calucalting flux and fitting it to power law curves.

Petersen shows how to derive FOM from experimental proton indeces SEI cross section . FOM is proportional to the limiting p-SEU cross section $\sigma_{pL}$ which can be used to calculate p-SEU rates and heavy ion SEU rates on devices. Using $FOM \propto \sigma_{pL}$ for devices with high LET leads to over estimation of p-SEU rates and underestimation of ehavy ion SEU rates.

# II. Heavy ION SEU Rates Using FOM 

## A. General Definition of the FOM for Heavy Ions

$$
FOM = \frac{\sigma_{HL}}{(L_{0.25})^2}
$$
where $\sigma_{HL}$ is limiting (saturation)} value of heavy ion cross section per bit with cm^2 per bit units. $L_{0.25}$ units $MeV \cdot cm^2/mg$ is the value in which $\sigma(L_{0.25}) = 0.25 \sigma_{HL}$ 

Using FOM for SEU rate was based on simplified LET spectrum and RPP geometry.
$$
	R = C \times FOM
$$
where $C$ is rate of Coeff given in upsets/bit-day. only depends on 
1. obit
2. shielding
3. ions
4. device hardness

heavy ion SEU rate $R_H$ calculated by
$$
R_H = \int_0^\infty f(L) \sigma_e(L)dL
$$
with $\sigma_e(L)$ the effective cross section (averages on all solid angles) . $f(L)$ is heavy ion differential ion flux spectrum in a given orbit for a given shielding

## B. Analytic Evaluation of the FOM  Computed form Heavy Ion Data
$$
R_H \propto FOM \propto^{-1} L_{0.25}  
$$
A simple $L_{0.25}^{-2}$ is not good enough. some relations for heavy ions whose $f(L) \propto L^{-3}$

calculating $R_H$ we find $f(L)$ using standard code like CREME96.a unit change must happen so the CREME96 output differential flux output is multiplied by $108573$ 

LET range for ions can be divided into sections in which $f(L)$ is presented by power laws. an example with Galactic Cosmic Rays GCRs, is fitted with three functions $f(L) = pL^{-n}$ for three domains. For low and high LET values the flux is much weaker and the SEU rates are expected to fall well below those calculated by equation $R = C \times FOM$ 

for calculations we take a disc with diameter $a$, thickness $c$ and high aspect ratio $c<< a$ with critical energy $\epsilon_c$. a exact chord length distribution was used to approximate. for beam at angle $\theta$ to ion track is $l = c/\cos\theta$ with cross section $\sigma_\theta = S\cos\theta$ with $S = \pi a^2/4$. Energy deposited is $\epsilon = \rho c L / \cos\theta$. 

Using $R_H = \int f(L) \sigma_e(L) dL$ and critical LET $L_c \geq L_{max}$ and $\cos\theta = L_{max}/L_c$:

$$\
R_H = \int_{cos\theta = L_{max}/ L_c}^{\theta = \theta_{max}}S\cos\theta \left[ \int_{l_c\cos\theta}^{L_{max}}pL^{-n}dL \right]d(-\cos\theta)
$$
where $L_c$ is the critical LET: $\epsilon_c = L_c\rho c$ . $\theta_{max}$ introduced since max chord length is $\approx a$ thus $\cos\theta_{max} \approx c/a$. For $L_c < L_{max}$ and $n > 3$

$$
R_H = \frac{p[(a/c)^{n-3}- 1]}{(n-1)(n-3)}\frac{S}{L_c^{n-1}} - \frac{p[(1-c/a)^2]}{2(n-1)}\frac{S}{L_{max^{n-1}}}
$$
modern devices may have  $a\approx c$ so the above turns into
$$
R_H = \frac{pS}{n-1}\left(\frac{1}{L_c^{n-1}} - \frac{1}{L_{max}^{n-1}}\right)
$$

## C. Discussion of Results

Increasing $L_c$ towards $L_{max}$ and even passing will decrease SEU rate much faster than expected from just FOM formula. To explain discrepancy, the higher effective LET of the heavy ions when hitting at grazing angles enables them to induce SEU. the other extreme is that of devices with $L_c \leq 1$. FOM model overestimates $R_H$


Google deep mind scarping data for prediction future.