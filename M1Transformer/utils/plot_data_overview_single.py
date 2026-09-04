"""All flux channels on a single axis, with solar-storm spans shaded."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

DATA = Path(__file__).resolve().parents[1] / "data" / "data.csv"
EVENTS = Path(__file__).resolve().parents[1] / "data" / "event_indices.txt"
CHANNELS = ["electron", "electron_high", "proton"]
LABELS = {"electron": "E150", "electron_high": "E300", "proton": "proton"}
COLORS = {"electron": "tab:blue", "electron_high": "tab:green", "proton": "tab:purple"}


def load_events(time):
    spans = []
    for line in EVENTS.read_text().splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        onset, end = int(cols[0]), int(cols[3])
        spans.append((time.iloc[onset], time.iloc[end]))
    return spans


def main():
    df = pd.read_csv(DATA, usecols=["time"] + CHANNELS, parse_dates=["time"])
    events = load_events(df["time"])

    fig, ax = plt.subplots(figsize=(16, 7))
    for channel in CHANNELS:
        ax.plot(
            df["time"], df[channel], lw=0.3,
            color=COLORS[channel], label=LABELS[channel],
        )

    for onset, end in events:
        ax.axvspan(
            onset, end, facecolor="tab:orange", alpha=0.3,
            edgecolor="darkred", lw=0.9,
        )

    ax.grid(alpha=0.3)
    ax.set_ylabel(r"$\ln(\mathrm{cm^2 \cdot s \cdot sr \cdot MeV})^{-1}$")
    ax.set_title(
        "Particle Flux Dataset from COSTEP-EPHIN on SOHO\n"
        f"{df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}"
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(
        facecolor="tab:orange", alpha=0.3, edgecolor="darkred", lw=0.9,
        label=f"solar storm (onset–end), {len(events)} total",
    ))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = DATA.parent / "data_overview_single.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
