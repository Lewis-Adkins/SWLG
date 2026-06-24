
import sklearn as sk
import pandas as df
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def Record_Overall_Results(a_df, a_batch_size, a_losses, a_model_name, a_graph):
    mean_electron_flux = (a_df["E300"] + a_df["E150"]) / 2
    
    panels = [
        (a_df["predicted"], "Predicted", "Predicted Proton Flux from >.25 MeV and >.67 MeV Electron Channels"),
        (a_df["E150"],      "E150",      "Original Proton Flux vs. >.25 MeV"),
        (a_df["E300"],      "E300",      "Original Proton Flux vs. >.67 MeV"),
        (mean_electron_flux, "E-Mean",   "Original Proton Flux vs. Mean Electron Flux"),
    ]


    if a_graph:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        fig.autofmt_xdate()
        fig.supylabel(r"$\ln( \text{cm}^2·\text{s}·\text{sr}·\text{MeV})^{-1}$")

        x = np.linspace(0, a_df.shape[0], len(a_losses)).round().astype(int)
        print(a_df.shape[0] // a_batch_size + 1,a_df.shape[0], len(x), len(a_losses))
        x[-1] = x[-1] - 1
        MSE_timestamps = a_df["Timestamp"].iloc[x]
    else:
        axes = [None] * len(panels)   # so the zip below still iterates

    results = {}

    for ax, (series, label, title) in zip(axes, panels):
        r2  = sk.metrics.r2_score(a_df["original"], series)
        mse = sk.metrics.mean_squared_error(a_df["original"], series)
        mae = sk.metrics.mean_absolute_error(a_df["original"], series)

        if label == "Predicted":
            results = {"r2": r2, "mse": mse, "mae": mae}

        if a_graph:
            ax.plot(a_df["Timestamp"], a_df["original"], label="Original", zorder=1)
            ax.plot(a_df["Timestamp"], series, label=label, zorder=2)
            ax.set_title(title)
            ax.annotate(f'R2: {r2:.4f}',  xy=(0.001, 0.001), xytext=(0.05, 0.07),
                        xycoords='axes fraction', bbox=dict(boxstyle="round", fc="0.8"))
            ax.annotate(f'MSE: {mse:.4f}', xy=(0.001, 0.001), xytext=(0.255, 0.07),
                        xycoords='axes fraction', bbox=dict(boxstyle="round", fc="0.8"))
            ax.annotate(f'MAE: {mae:.4f}', xy=(0.001, 0.001), xytext=(0.505, 0.07),
                        xycoords='axes fraction', bbox=dict(boxstyle="round", fc="0.8"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.legend(loc='lower right')

    if a_graph:
        print(MSE_timestamps, a_losses)
        axes[0].plot(MSE_timestamps, a_losses.detach().cpu().numpy(), label="MSE")
        axes[0].legend(loc='lower right')
        plt.savefig('results/plots/' + str(a_model_name) + '.png')
        plt.close(fig)

    return results