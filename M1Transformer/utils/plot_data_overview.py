"""Plot the raw flux channels (electron, electron_high, proton) over the full record."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

DATA = Path(__file__).resolve().parents[1] / "data" / "data.csv"
EVENTS = Path(__file__).resolve().parents[1] / "data" / "event_indices.txt"
CHANNELS = ["electron", "electron_high", "proton"]
LABELS = {"electron": "E150", "electron_high": "E300", "proton": "proton"}


def load_events(time):
    """Return (onset, end) timestamp pairs from the onset/end columns of each row."""
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

    fig, axes = plt.subplots(
        len(CHANNELS), 1, figsize=(16, 9), sharex=True, sharey=True
    )
    for ax, channel in zip(axes, CHANNELS):
        ax.plot(df["time"], df[channel], lw=0.3)
        ax.grid(alpha=0.3)
        for onset, end in events:
            ax.axvspan(
                onset, end, facecolor="tab:orange", alpha=0.3,
                edgecolor="darkred", lw=0.9,
            )
        ax.text(
            0.005, 0.95, LABELS[channel], transform=ax.transAxes,
            va="top", ha="left", fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
        )

    fig.supylabel(r"$\ln(\mathrm{cm^2 \cdot s \cdot sr \cdot MeV})^{-1}$")

    storm_patch = Patch(
        facecolor="tab:orange", alpha=0.3, edgecolor="darkred", lw=0.9,
        label=f"each vertical orange band = one solar storm (onset–end), "
        f"{len(events)} total",
    )
    fig.legend(handles=[storm_patch], loc="lower center", ncol=1, frameon=False)

    axes[0].set_title(
        "Particle Flux Dataset from COSTEP-EPHIN on SOHO\n"
        f"{df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}"
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = DATA.parent / "data_overview.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
