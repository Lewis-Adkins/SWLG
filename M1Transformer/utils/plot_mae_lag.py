"""
No-Phase vs. With-Phase Inputs comparison chart.

Paired (touching) bars for methods with both variants confirmed
(Torres NN, Torres RNN, Linear Reg, Zero-Init, Sinusoidal).
Solo bar for methods with only no-phase confirmed (RoPE).

Edit the dictionaries below to update values, add/remove methods, etc.
Each value is a (mean, std) tuple. Use std=0 for deterministic methods
(Linear Regression -- no seed variance).
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- Methods with BOTH no-phase and with-phase data ----
paired_methods = ["Torres NN", "Torres RNN", "Linear Reg", "Zero-Init", "Sinusoidal"]

# ---- Methods with ONLY no-phase data (no matching phase-run yet) ----
solo_methods = ["Persistence", "Posner", "RoPE"]

all_methods = paired_methods + solo_methods

# ---- Bar colors per method (edit to change palette) ----
colors_nophase = {
    "Persistence": "#999999",
    "Posner":      "#8172B2",
    "Torres NN":   "#55A868",
    "Torres RNN":  "#4C72B0",
    "Linear Reg":  "#D9A441",
    "Zero-Init":   "#F4B183",
    "Sinusoidal":  "#C44E52",
    "RoPE":        "#A64D79",
}

# =================================================================
# DATA -- (mean, std) tuples
# =================================================================

# ---- MAE, no-phase ----
mae_t6_nophase  = {"Persistence": (0.419, 0), "Posner": (1.531, 0),
                    "Torres NN": (0.441, 0.018), "Torres RNN": (0.379, 0.025),
                    "Linear Reg": (0.438, 0),
                    "Zero-Init": (0.349, 0.011), "Sinusoidal": (0.350, 0.011),
                    "RoPE": (0.352, 0.028)}
mae_t12_nophase = {"Persistence": (0.732, 0), "Posner": (1.586, 0),
                    "Torres NN": (0.690, 0.037), "Torres RNN": (0.599, 0.016),
                    "Linear Reg": (0.669, 0),
                    "Zero-Init": (0.556, 0.015), "Sinusoidal": (0.576, 0.039),
                    "RoPE": (0.596, 0.027)}

# ---- O2P Lag, no-phase ----
lag_t6_nophase  = {"Persistence": (6.000, 0), "Posner": (4.000, 0),
                    "Torres NN": (4.444, 0.396), "Torres RNN": (5.222, 0.396),
                    "Linear Reg": (6.500, 0),
                    "Zero-Init": (5.322, 0.184), "Sinusoidal": (5.444, 0.241),
                    "RoPE": (5.411, 0.282)}
lag_t12_nophase = {"Persistence": (12.000, 0), "Posner": (6.722, 0),
                    "Torres NN": (9.600, 0.422), "Torres RNN": (9.722, 0.798),
                    "Linear Reg": (11.778, 0),
                    "Zero-Init": (10.078, 0.204), "Sinusoidal": (10.322, 0.510),
                    "RoPE": (10.567, 0.275)}

# ---- MAE, with phase ----
mae_t6_phase  = {"Torres NN": (0.475, 0.012), "Torres RNN": (0.405, 0.033),
                  "Linear Reg": (0.412, 0),
                  "Zero-Init": (0.335, 0.008), "Sinusoidal": (0.347, 0.007)}
mae_t12_phase = {"Torres NN": (0.653, 0.036), "Torres RNN": (0.648, 0.041),
                  "Linear Reg": (0.626, 0),
                  "Zero-Init": (0.564, 0.012), "Sinusoidal": (0.575, 0.017)}

# ---- O2P Lag, with phase ----
lag_t6_phase  = {"Torres NN": (5.289, 0.512), "Torres RNN": (4.589, 0.655),
                  "Linear Reg": (5.556, 0),
                  "Zero-Init": (4.978, 0.386), "Sinusoidal": (4.533, 0.745)}
lag_t12_phase = {"Torres NN": (7.956, 0.523), "Torres RNN": (8.733, 0.577),
                  "Linear Reg": (10.333, 0),
                  "Zero-Init": (10.322, 0.323), "Sinusoidal": (9.022, 1.405)}


# =================================================================
# Plotting
# =================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))


def plot_panel(ax, nophase_d, phase_d, title, ylabel):
    """Plot one panel: paired (no-phase/with-phase) bars where available,
    solo bars otherwise."""
    x = np.arange(len(all_methods))
    width = 0.35

    for i, m in enumerate(all_methods):
        v_np, e_np = nophase_d[m]
        yerr_np = e_np if e_np > 0 else None

        if m in phase_d:
            # paired bars, touching (no gap)
            ax.bar(i - width / 2, v_np, width, yerr=yerr_np, capsize=4,
                   color=colors_nophase[m], edgecolor="black",
                   label="No Phase" if i == 0 else "")

            v_p, e_p = phase_d[m]
            yerr_p = e_p if e_p > 0 else None
            ax.bar(i + width / 2, v_p, width, yerr=yerr_p, capsize=4,
                   color=colors_nophase[m], edgecolor="black", hatch="....",
                   label="With Phase" if i == 0 else "")

            ax.text(i - width / 2, v_np + 0.02 * max(v_np, 1), f"{v_np:.3f}",
                    ha="center", fontsize=6.5)
            ax.text(i + width / 2, v_p + 0.02 * max(v_p, 1), f"{v_p:.3f}",
                    ha="center", fontsize=6.5)
        else:
            # solo bar -- no phase-included run available for this method
            ax.bar(i, v_np, width, yerr=yerr_np, capsize=4,
                   color=colors_nophase[m], edgecolor="black")
            ax.text(i, v_np + 0.02 * max(v_np, 1), f"{v_np:.3f}",
                    ha="center", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")


plot_panel(axes[0, 0], mae_t6_nophase,  mae_t6_phase,  "MAE at t=6 (30 min)",      "Rising-phase MAE (ln units)")
plot_panel(axes[0, 1], mae_t12_nophase, mae_t12_phase, "MAE at t=12 (60 min)",     "Rising-phase MAE (ln units)")
plot_panel(axes[1, 0], lag_t6_nophase,  lag_t6_phase,  "O2P Lag at t=6 (30 min)",  "Lag (5-min timesteps)")
plot_panel(axes[1, 1], lag_t12_nophase, lag_t12_phase, "O2P Lag at t=12 (60 min)", "Lag (5-min timesteps)")

axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle(
    "No-Phase vs. With-Phase Inputs, Comparing MAE and O2P Lag\n",
    fontsize=12, fontweight="bold", y=1.03,
)
fig.tight_layout()

fig.savefig("mae_lag_phase_comparison.png", dpi=150, bbox_inches="tight")
# fig.savefig("mae_lag_phase_comparison.svg", format="svg", bbox_inches="tight")  # uncomment for vector output
print("Saved mae_lag_phase_comparison.png")