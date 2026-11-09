from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from lib.utils.events import read_event_timestamps

# Font sizes for the figures saved into the LaTeX paper (11 pt text, 12 pt labels and titles).
# Used as a decorator: @matplotlib.rc_context(PAPER_RC)
PAPER_RC = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11,
            'ytick.labelsize': 11, 'legend.fontsize': 11, 'figure.titlesize': 12}


class PlotStuff:
    # Takes an Analysis object and reads its settings (metrics, times, baseline names,
    # results folder) from it, so the two classes never disagree.
    # analysis=None is enough for the data overview plots (plot_flux_series etc.), which only read data/.

    # ---------- AGU journal figure style (Space Weather / JGR: Space Physics, agujournal class) ----------
    # AGU graphic requirements: text ~8 pt at final print size (sub/superscripts ~6 pt); Arial, Helvetica, Times
    # or Symbol, embedded; lines >= 0.5 pt; one column 50-85 mm, two columns 105-170 mm, height <= 228 mm;
    # PDF/EPS preferred (raster 300-600 ppi, RGB); no "Figure 1"/caption inside the graphic; panels (a), (b), ...
    # Save at the final size (AGU_ONE_COL / AGU_TWO_COL) and include in LaTeX at \textwidth (two-column) or
    # \columnwidth (one-column) without rescaling, so 8 pt here is 8 pt in print.
    # Use inside a method as: with matplotlib.rc_context(self.AGU_RC): ...
    MM = 1 / 25.4                 # inches per mm
    AGU_ONE_COL = 85 * MM         # 3.35 in
    AGU_TWO_COL = 170 * MM        # 6.69 in
    AGU_MAX_H = 228 * MM          # 8.98 in
    AGU_RC = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'patch.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'grid.linewidth': 0.5,
        'legend.frameon': False,
        'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,          # embed TrueType fonts
    }

    @classmethod
    def agu_apply(cls) -> None:
        # Set the AGU style globally (inside plotting methods prefer matplotlib.rc_context(self.AGU_RC))
        matplotlib.rcParams.update(cls.AGU_RC)

    @staticmethod
    def agu_panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
        # AGU-style multipart label: bold lowercase letter in parentheses at the top left of the axes
        ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, weight='bold')

    def __init__(self, analysis=None, show: bool = False):
        self.analysis = analysis
        self.show     = show   # open the histogram booklet window (blocks until closed); PNGs are saved either way
        # "Bright" palette from simplifiedsciencepublishing.com, in the order listed there.
        self.palette  = ['#003a7d', '#008dff', '#ff73b6', '#c701ff',
                         '#4ecb8d', '#ff9d3a', '#f9e858', '#d83034']
        self.best_c   = self.palette[4]   # green border on the info box of the best model
        self.sep_c    = self.palette[5]   # orange shading for SEP events in the data overview plots
        self.baseline_c = self.palette[5]    # MAE/MSE plots: baselines orange ...
        self.champion_c = self.palette[1]    # ... champions blue
        # Each event gets its own colour + marker (8 colours x 5 shapes = 40 styles for the 39 events),
        # fixed by event number so an event looks the same in every plot.
        self.event_markers = ['o', 's', '^', 'D', 'v']

        # What "best" means for each metric and its per-event folder, from the one table in Analysis.metric_info
        metric_info = analysis.metric_info if analysis is not None else {}
        self.metric_best = {m: info['best'] for m, info in metric_info.items()}

        # Metric histograms: one folder per metric, e.g. metrics_analysis/hists/phases/mae/
        self.metric_folders = {m: info['folder'] for m, info in metric_info.items()}
        self.phase_folders  = {'phases': 'phases', 'nophases': 'no_phases'}
        # Posner predicts flux in steps (bins), so its predicted peak/threshold often lands on the true one by
        # construction and its lags are mostly exactly 0 even though it over-predicts. It can't win a lag metric.
        self.lag_disqualified = {'posner'}
        self.lag_note = ('posner is disqualified from the lag "best" counts:\n'
                         'it over-predicts in steps, so its lags are\nmostly exactly 0 without forecasting well')

    @staticmethod
    def plot_event_forecast(times_o2p: list, targets_o2p: dict, predictions_o2p: dict, electron_o2p: dict,
                            electron_high_o2p: dict, path: str | None, event_number: int, display: bool = False) -> None:
        # Torres's per-event plot, called from BaselineModel.evaluate for every run: actual vs predicted proton
        # flux plus both electron channels, from 36 steps before onset to the peak, with the ln(10) line.
        # Saved as <run folder>/event<N>.png.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_ylabel('ln(Flux (/cc/s/sr))')
        ax.plot(list(targets_o2p.values()), '-b', label='Actual proton')
        ax.plot(list(predictions_o2p.values()), '-r', label='Predicted proton')
        ax.plot(list(electron_o2p.values()), '-m', label='Electron')
        ax.plot(list(electron_high_o2p.values()), '-y', label='High-energy electron')
        ax.plot([np.log(10)] * len(targets_o2p.values()), '--k')

        # Set x-axis to timestamps
        diff = int(ax.get_xticks()[1] - ax.get_xticks()[0])
        time_labels = [time[time.index('T') + 1: time.index('.')] for time in times_o2p[::diff]]
        if len(time_labels) == len(ax.get_xticks()) - 2:
            ax.set_xticks(ax.get_xticks()[1:-1])
        else:
            ax.set_xticks(ax.get_xticks()[1:-2])
        ax.set_xticklabels(time_labels, rotation='vertical')

        # Caption with date
        date_start = times_o2p[0][:times_o2p[0].index('T')]
        date_end = times_o2p[-1][:times_o2p[-1].index('T')]
        if date_start != date_end:
            ax.set_xlabel(f'Event from {date_start} to {date_end}')
        else:
            ax.set_xlabel(f'Event on {date_start}')

        fig.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
        fig.tight_layout()
        if path:
            plt.savefig(f'{path}/event{event_number}.png')
        if display:
            plt.show()
        plt.close()

    @staticmethod
    @matplotlib.rc_context(PAPER_RC)
    def plot_lag_diagram(config: dict, event_timestamps: list, model: str = 'sin', phases: str = 'phases',
                         dataset: int = 0, t: int = 6, run: int = 0, event: int = 1, x_max: int | None = 100,
                         lag_at: int | None = 40,
                         display: bool = False) -> None:
        # Torres's lag/error diagram for one real event, from a run's saved predictions.txt (event is 1-based,
        # as in results.txt). Saved in the paper figs (evaluation_criteria/equations) as lag_diagram_<model>_t+<T>_<phases>_dataset<N>_run<RR>_event<E>.png
        import matplotlib.pyplot as plt
        from core.rescore import test_targets
        from lib.torres.stats import x_axis_error
        from models.baseline_model import BaselineModel   # local: baseline_model imports PlotStuff

        run_folder = (Path(config["admin"]["results_folder_path"]) / model / phases / f"dataset_{dataset}" /
                      f"t+{t}" / f"{model}-{run:02d}")
        data = pd.read_csv(Path(config["admin"]["data_folder_path"]) / f"data_{dataset}.csv")
        targets = test_targets(data, t, config["data"]["train_split"])
        predictions = dict(zip(targets.keys(), np.loadtxt(run_folder / "predictions.txt")))
        scorer = BaselineModel(config)
        onset, _, peak = scorer.test_events(event_timestamps, targets)[event - 1][:3]

        # Window: 36 steps (3 h) of background before onset, up to the peak
        times = list(targets.keys())
        start = times[times.index(onset) - 36]
        y_true = np.array(list(scorer.slice_dictionary_keys(targets, start, peak).values()))
        y_pred = np.array(list(scorer.slice_dictionary_keys(predictions, start, peak).values()))
        i_onset = 36

        # Lag = shift with the lowest MAE over onset..peak (same as the O2P lag in results.txt)
        lag = int(x_axis_error(y_true[i_onset:], y_pred[i_onset:], False))
        x = np.arange(len(y_true))
        y_shift = y_pred[lag:]

        fig, ax = plt.subplots(figsize=(7, 5))
        palette = PlotStuff().palette
        true_c, pred_c, err_c, lag_c = palette[0], palette[7], palette[3], palette[4]
        ax.plot(x, y_true, '-', color=true_c, label='actual proton')
        ax.plot(x, y_pred, '-', color=pred_c, label='predicted proton')
        ax.axhline(np.log(10), color='#0b0b0b', ls=':', lw=1, label='ln(10) threshold')
        ax.plot(x[:len(y_shift)], y_shift, '--', color=pred_c, label='shifted pred. proton\nwith min. MAE')

        # Error: largest vertical gap between actual and predicted after onset
        i_err = i_onset + int(np.argmax(np.abs(y_true[i_onset:] - y_pred[i_onset:])))
        ax.annotate('', xy=(i_err, y_pred[i_err]), xytext=(i_err, y_true[i_err]),
                    arrowprops=dict(arrowstyle='<->', color=err_c, lw=0.8))
        ax.text(i_err + 2, (y_true[i_err] + y_pred[i_err]) / 2, 'Error', color=err_c, va='center')

        # Lag: horizontal arrow from the shifted curve to the predicted curve, starting at step lag_at
        # (None = halfway up the rise)
        mid = (y_true[i_onset] + y_true.max()) / 2
        i_lag = lag_at if lag_at is not None else \
            next((i for i in range(i_onset, len(y_shift)) if y_shift[i] >= mid), i_onset)
        # +1 step on the predicted end so the arrowhead clears the red line instead of sitting on it
        ax.annotate('', xy=(i_lag + lag + 1, y_shift[i_lag]), xytext=(i_lag, y_shift[i_lag]),
                    arrowprops=dict(arrowstyle='<->', color=lag_c, lw=0.8))
        ax.annotate('lag', xy=(i_lag + lag / 2, y_shift[i_lag]),
                    xytext=(i_lag + lag + 15, y_shift[i_lag] - 1), color=lag_c,
                    arrowprops=dict(arrowstyle='-', color=lag_c, lw=0.5))

        ax.annotate('Onset', xy=(i_onset, y_true[i_onset]), xytext=(i_onset + 8, y_true[i_onset] - 0.4),
                    arrowprops=dict(arrowstyle='-', lw=0.5))
        if x_max is None or x_max >= len(x) - 1:
            ax.text(len(x) - 1, y_true[-1], 'Peak', ha='right', va='bottom')
        if x_max is not None:
            ax.set_xlim(0, x_max)   # zoom on the rise so the lag arrow is visible (None = whole event)

        ax.set_xlabel(f'Time (5-min steps from {start[:16]})')
        ax.set_ylabel('ln(Flux)')
        ax.set_title(f'{model} t+{t} ({phases}), dataset {dataset}, run {run:02d}, event {event}')
        ax.legend(loc='upper left')
        fig.tight_layout()
        out_dir = Path(r'C:\Users\Lewis\Documents\SWLG-Paper\figs\evaluation\time_intensity_profile'
                       r'\evaluation_criteria\equations')
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f'lag_diagram_{model}_t+{t}_{phases}_dataset{dataset}_run{run:02d}_event{event}.png', dpi=150)
        if display:
            plt.show()
        plt.close(fig)

    def _event_style(self, event: int) -> dict:
        return dict(color=self.palette[event % len(self.palette)],
                    marker=self.event_markers[(event // len(self.palette)) % len(self.event_markers)])

    def _event_title(self, e: str) -> str:
        # e.g. "event 22 (onset 2001-01-28T17:07:27.264), in datasets 1, 3, 8"
        a = self.analysis
        datasets = sorted({int(ds.split('_')[1]) for t in a.times for ds, events in a.event_slices[t].items() if e in events})
        return f"event {int(e[1:])} (onset {a.event_onsets[e]}), in datasets {', '.join(map(str, datasets))}"

    def _event_label(self, event: int) -> str:
        # e.g. "E22 2001-01-28" (1-based line number in event_timestamps.txt + onset date)
        return f"E{event + 1} {self.analysis.event_onsets[event][:10]}"

    def _info_box(self, best: bool) -> dict:
        # Wheat box like the HW plots; a thick green border marks the best model.
        # Transparency goes on the face only, so the border stays solid and the box stays readable over bars.
        return dict(boxstyle='round', facecolor=to_rgba('wheat', 0.85), pad=0.5,
                    edgecolor=self.best_c if best else 'gray', linewidth=3 if best else 1)

    def _best_models(self, means: pd.DataFrame, metric: str) -> dict:
        # {(phases, t): best model_type} for one metric, comparing every model (baselines and champions).
        rule = self.metric_best.get(metric, 'max')
        col = means[metric].dropna()
        score = {'max': -col, 'min': col, 'zero': col.abs()}[rule]
        best = {}
        for (model_type, phases, t), value in score.items():
            if (phases, t) not in best or value < score[(best[(phases, t)], phases, t)]:
                best[(phases, t)] = model_type
        return best

    def _load_summary(self, prefix: str, name: str) -> pd.DataFrame:
        # Reads results/<metrics folder of the bootstrap mode>/{prefix}_{name}.csv, indexed by the mode's
        # bootstrap keys: (model_type, phases, t) for 'event_avg', plus event for 'events'.
        return pd.read_csv(self.analysis.csv_path(prefix, name)).set_index(self.analysis.bootstrap_keys)

    def _plot_metric_histograms(self, data: tuple[pd.DataFrame], booklet_title: str, centered: bool) -> None:
        # One PNG per (phases, metric): rows = models, columns = t+6 / t+12, saved next to that metric's per-event
        # plots as metrics_analysis/hists/<phases|no_phases>/<metric>/events_avg.png (e.g. .../phases/mae/events_avg.png).
        # It reads the CSVs in metrics_analysis/csv/<baseline|champion>/event_avg/.
        # Each panel gets: mean line, dashed + shaded mean ± SE, a line at mean + SB, the 95% percentile CI,
        # and an info box (green border = best model for that metric, horizon and phases).
        # Mean, SE and SB come from the CSVs written by Analysis.get_mean_bootstrap_estimates,
        # get_bootstrap_standard_error and get_bootstrap_standard_bias.
        # centered=True means the data are diffs from the mean, so the mean is drawn at 0 (saved as events_avg_diffs.png).
        a       = self.analysis
        metrics = a.desired_results_metrics
        times   = a.times
        values  = pd.concat(data, ignore_index=True)
        models  = [mt for mt in a.baseline_names + a.champion_names if mt in set(values['model_type'])]
        means   = pd.concat([self._load_summary(prefix, 'mean_bootstrap_estimates') for prefix in ('baseline', 'champion')])
        ses     = pd.concat([self._load_summary(prefix, 'bootstrap_standard_errors') for prefix in ('baseline', 'champion')])
        sbs     = pd.concat([self._load_summary(prefix, 'bootstrap_standard_bias') for prefix in ('baseline', 'champion')])
        # Best model per metric (green box), compared across baselines and champions together.
        best    = {dm: self._best_models(means, dm) for dm in metrics}
        groups  = {k: g for k, g in values.groupby(['model_type', 'phases', 't'])}
        bar_c, mean_c, se_c, sb_c, ci_c = self.palette[:5]

        pages = []
        for phases in a.phases:
            for dm in metrics:
                name = self.metric_folders.get(dm, dm.lower().replace(' ', '_'))
                out_dir = a.results_folder / a.hists_folder / self.phase_folders[phases] / name
                out_dir.mkdir(parents=True, exist_ok=True)
                # Plain Figure (not plt.subplots) so pyplot doesn't open its own window.
                fig = Figure(figsize=(12, 3 * len(models)), dpi=100)
                axes = fig.subplots(len(models), len(times), squeeze=False)
                # Train Time is stored in seconds (see Analysis.pull_metrics_from_results_file).
                unit = ' (s)' if dm == 'Train Time' else ''
                fig.suptitle(f'{phases}: {dm}{unit}, one value per dataset (average over its events)')

                for row, model_type in enumerate(models):
                    for col, t in enumerate(times):
                        ax = axes[row, col]
                        key = (model_type, phases, t)
                        g = groups.get(key)
                        v = g[dm].dropna() if g is not None else pd.Series(dtype=float)
                        ax.set_title(f'{model_type}, {t} (n={len(v)})', fontsize=10)
                        ax.set_ylabel('count')
                        is_best = best[dm].get((phases, t)) == model_type

                        # Every value identical (e.g. persistence train time): nothing to bin.
                        if v.nunique() <= 1:
                            ax.text(0.5, 0.5, f'all values = {v.iloc[0]:.3f}' if len(v) else 'no data',
                                    ha='center', va='center', transform=ax.transAxes,
                                    bbox=self._info_box(is_best and len(v) > 0))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            continue

                        # numpy's 'auto' bin count, but never fewer than 4 bins.
                        n_bins = max(4, len(np.histogram_bin_edges(v, bins='auto')) - 1)
                        ax.hist(v, bins=n_bins, color=bar_c, edgecolor='white')

                        mean = 0.0 if centered else means.loc[key, dm]
                        se = ses.loc[key, dm]
                        sb = sbs.loc[key, dm]
                        # 95% percentile CI: 2.5th / 97.5th percentiles of the values.
                        ci_lo, ci_hi = np.percentile(v, [2.5, 97.5])

                        top = ax.get_ylim()[1]
                        ax.axvspan(mean - se, mean + se, color=se_c, alpha=0.15, zorder=0)
                        ax.vlines([mean - se, mean + se], 0, top, colors=se_c, linestyles='--', linewidth=2, label='mean ± SE')
                        ax.vlines(mean, 0, top, colors=mean_c, linestyles='--', linewidth=2, label='mean')
                        ax.vlines(mean + sb, 0, top, colors=sb_c, linewidth=2, label='mean + SB')
                        ax.vlines([ci_lo, ci_hi], 0, top, colors=ci_c, linestyles=':', linewidth=2.5, label='95% CI (percentile)')
                        ax.set_ylim(0, top)

                        # Same info box as the other plots; green border = best mean for this metric.
                        text = (f"mean = {mean:.3f}\n"
                                f"SE = {se:.3f}\n"
                                f"SB = {sb:.3f}\n"
                                f"95% CI: ({ci_lo:.3f}, {ci_hi:.3f})")
                        ax.text(0.97, 0.95, text, fontsize=8, ha='right', va='top', transform=ax.transAxes,
                                bbox=self._info_box(is_best))

                # One shared legend per page instead of one in every panel.
                handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat
                                        if ax.get_legend_handles_labels()[0]), ([], []))
                fig.legend(handles, labels, loc='upper right', ncol=4)
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                fig.savefig(out_dir / f"events_avg{'_diffs' if centered else ''}.png", dpi=120)
                pages.append((f'{phases}, {dm}', fig))

        if self.show:
            self._show_booklet(pages, booklet_title)

    @matplotlib.rc_context(PAPER_RC)
    def plot_single_events_avg(self, mean_estimate_data: tuple[pd.DataFrame], model_type: str = 'sin',
                               phases: str = 'phases', t: str = 't+6', metric: str = 'MAE',
                               out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\bootstrap') -> None:
        # One panel of events_avg.png on its own (default: sin, t+6, phases, MAE), drawn the same way.
        # Needs the 'event_avg' bootstrap mode. Saved as <out_dir>/<metric>_<model>_<t>_<phases>.png
        bar_c, mean_c, se_c, sb_c, ci_c = self.palette[:5]
        values = pd.concat(mean_estimate_data, ignore_index=True)
        means = pd.concat([self._load_summary(p, 'mean_bootstrap_estimates') for p in ('baseline', 'champion')])
        ses   = pd.concat([self._load_summary(p, 'bootstrap_standard_errors') for p in ('baseline', 'champion')])
        sbs   = pd.concat([self._load_summary(p, 'bootstrap_standard_bias') for p in ('baseline', 'champion')])
        key = (model_type, phases, t)
        v = values[(values['model_type'] == model_type) & (values['phases'] == phases) & (values['t'] == t)][metric].dropna()
        is_best = self._best_models(means, metric).get((phases, t)) == model_type

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.subplots()
        ax.set_title(f'{model_type}, {t} (n={len(v)})')
        ax.set_ylabel('count')
        ax.set_xlabel(metric)

        n_bins = max(4, len(np.histogram_bin_edges(v, bins='auto')) - 1)
        ax.hist(v, bins=n_bins, color=bar_c, edgecolor='white')
        mean, se, sb = means.loc[key, metric], ses.loc[key, metric], sbs.loc[key, metric]
        ci_lo, ci_hi = np.percentile(v, [2.5, 97.5])

        top = ax.get_ylim()[1]
        ax.axvspan(mean - se, mean + se, color=se_c, alpha=0.15, zorder=0)
        ax.vlines([mean - se, mean + se], 0, top, colors=se_c, linestyles='--', linewidth=2, label='mean ± SE')
        ax.vlines(mean, 0, top, colors=mean_c, linestyles='--', linewidth=2, label='mean')
        ax.vlines(mean + sb, 0, top, colors=sb_c, linewidth=2, label='mean + SB')
        ax.vlines([ci_lo, ci_hi], 0, top, colors=ci_c, linestyles=':', linewidth=2.5, label='95% CI (percentile)')
        ax.set_ylim(0, top)
        text = (f"mean = {mean:.3f}\n"
                f"SE = {se:.3f}\n"
                f"SB = {sb:.3f}\n"
                f"95% CI: ({ci_lo:.3f}, {ci_hi:.3f})")
        ax.text(0.97, 0.95, text, ha='right', va='top', transform=ax.transAxes,
                bbox=self._info_box(is_best))
        ax.legend(loc='upper left')
        fig.tight_layout()

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{metric.lower().replace(' ', '_')}_{model_type}_{t}_{phases}.png", dpi=150)

    def _show_booklet(self, pages: list[tuple[str, Figure]], booklet_title: str) -> None:
        # One scrollable window with Prev/Next buttons (or Left/Right arrow keys) to flip pages.
        # Blocks until the window is closed.
        root = tk.Tk()

        nav = tk.Frame(root)
        nav.pack(side='top', fill='x')
        prev_btn = tk.Button(nav, text='◀ Prev', width=10)
        next_btn = tk.Button(nav, text='Next ▶', width=10)
        page_label = tk.Label(nav, font=('Segoe UI', 11, 'bold'))
        prev_btn.pack(side='left', padx=5, pady=5)
        next_btn.pack(side='right', padx=5, pady=5)
        page_label.pack(side='top', pady=5)

        body = tk.Frame(root)
        body.pack(side='top', fill='both', expand=True)
        canvas = tk.Canvas(body)
        scrollbar = tk.Scrollbar(body, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor='nw')

        state = {'i': 0}

        def show(i: int) -> None:
            state['i'] = i % len(pages)
            title, fig = pages[state['i']]
            for child in frame.winfo_children():
                child.destroy()
            fig_canvas = FigureCanvasTkAgg(fig, master=frame)
            fig_canvas.draw()
            NavigationToolbar2Tk(fig_canvas, frame).pack(side='top', fill='x')
            fig_canvas.get_tk_widget().pack()
            frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.yview_moveto(0)
            page_label.config(text=f'{title}   ({state["i"] + 1} / {len(pages)})')
            root.title(f'{booklet_title} - {title}')

        prev_btn.config(command=lambda: show(state['i'] - 1))
        next_btn.config(command=lambda: show(state['i'] + 1))
        root.bind('<Left>', lambda e: show(state['i'] - 1))
        root.bind('<Right>', lambda e: show(state['i'] + 1))
        canvas.bind_all('<MouseWheel>', lambda e: canvas.yview_scroll(int(-e.delta / 120), 'units'))

        show(0)

        # Window fits the figure's width; height is capped at 90% of the screen.
        width = frame.winfo_reqwidth() + scrollbar.winfo_reqwidth()
        height = min(frame.winfo_reqheight() + nav.winfo_reqheight(), int(root.winfo_screenheight() * 0.9))
        root.geometry(f'{width}x{height}+0+0')

        root.protocol('WM_DELETE_WINDOW', lambda: (root.quit(), root.destroy()))
        root.mainloop()

    # estimator names the statistic used (e.g. 'mean', 'median'); it goes into the
    # booklet title and the PNG filenames, e.g. bootstrap_mean_estimates_posner_phases.png
    def plot_diffs(self, mean_diff_data: tuple[pd.DataFrame], estimator: str = 'mean')-> None:
        self._plot_metric_histograms(mean_diff_data, booklet_title=f'Bootstrap {estimator} diffs', centered=True)

    def plot_mean_estimates(self, mean_estimate_data: tuple[pd.DataFrame], estimator: str = 'mean')-> None:
        self._plot_metric_histograms(mean_estimate_data, booklet_title=f'Bootstrap {estimator} estimates', centered=False)

    # ---------- true vs predicted ----------

    def _true_vs_pred_stats(self, pred_data: tuple[dict], mt: str, t: str, p: str) -> dict | None:
        # Pools every dataset for one (model, horizon, phases): the points are each dataset's
        # run-mean prediction vs its true protons, and every regression value is the bootstrap
        # mean across all datasets and runs. Returns None if the model has no data for this key.
        record = next((d for d in pred_data if p in d['data'].get(mt, {}).get(t, {})), None)
        if record is None:
            return None
        datasets = sorted(record['data'][mt][t][p], key=lambda ds: int(ds.split('_')[1]))

        def pooled_mean(name, idx=None):
            vals = [record[name][mt][t][p][ds] for ds in datasets]
            if idx is not None:
                vals = [v[idx] for v in vals]
            return float(np.mean(np.concatenate([np.ravel(v) for v in vals])))

        x = np.concatenate([record['data'][mt][t][p][ds].mean(axis=1) for ds in datasets])
        y = np.concatenate([self.analysis.true_protons[t][int(ds.split('_')[1])] for ds in datasets])

        # Which event each point belongs to, and each event's R^2 averaged over every dataset and run it appears in
        point_event = np.concatenate([self.analysis.event_ids[t][int(ds.split('_')[1])] for ds in datasets])
        r2_by_event = {}
        for ds in datasets:
            ids = np.unique(self.analysis.event_ids[t][int(ds.split('_')[1])])
            for e, row in zip(ids, record['event_r2'][mt][t][p][ds]):     # rows follow np.unique(ids)
                r2_by_event.setdefault(int(e), []).append(row)
        event_r2 = {e: float(np.nanmean(np.concatenate(rows))) if np.isfinite(np.concatenate(rows)).any() else np.nan
                    for e, rows in r2_by_event.items()}

        s = {name: pooled_mean(name) for name in ('b0', 'b1', 'b1_std_err', 'b0_std_err', 'mean', 'ssx', 's2', 't_crit', 'r2', 'r2_event_mean')}
        s['ci_b1'] = (pooled_mean('ci', 0), pooled_mean('ci', 1))
        s['ci_b0'] = (pooled_mean('ci_b0', 0), pooled_mean('ci_b0', 1))

        # PI lines share the fit's slope; their intercepts are b0 -/+ t*s (the 1/n and
        # (x - mean)^2 / ssx terms are tiny with ~117k points, but the exact formula is used for coverage).
        half = s['t_crit'] * np.sqrt(s['s2'])
        s['pi_b0'] = (s['b0'] - half, s['b0'] + half)
        pi_lo, pi_hi = self.analysis.get_prediction_predictions_intervals(
            x, s['mean'], s['ssx'], s['s2'], s['t_crit'], s['b0'], s['b1'])
        s['pi_cover'] = float(np.mean((y >= pi_lo) & (y <= pi_hi)))
        s['x'], s['y'], s['point_event'], s['event_r2'] = x, y, point_event, event_r2
        return s

    def _draw_true_vs_pred(self, ax, s: dict, title: str, fontsize: int = 10, best: bool = False) -> None:
        point_c, fit_c, ci_c, pi_c = self.palette[1], self.palette[0], self.palette[3], self.palette[7]
        x, y = s['x'], s['y']
        xs = np.linspace(x.min(), x.max(), 200)

        if 'point_event' in s:
            # One colour + marker per event (see the figure legend for which is which and each event's R^2)
            for e in np.unique(s['point_event']):
                in_event = s['point_event'] == e
                style = self._event_style(int(e))
                ax.scatter(x[in_event], y[in_event], s=10, alpha=0.5, color=style['color'], marker=style['marker'],
                           linewidths=0, rasterized=True)
        else:
            ax.scatter(x, y, s=10, alpha=0.5, color=point_c, linewidths=0, rasterized=True)
        ax.plot(xs, s['b1'] * xs + s['b0'], color=fit_c, linewidth=2, label='fit')
        ax.plot(xs, s['ci_b1'][0] * xs + s['b0'], '--', color=ci_c, linewidth=1.5, label=r'95% CI $\beta_1$')
        ax.plot(xs, s['ci_b1'][1] * xs + s['b0'], '--', color=ci_c, linewidth=1.5)
        pi_lo, pi_hi = self.analysis.get_prediction_predictions_intervals(
            xs, s['mean'], s['ssx'], s['s2'], s['t_crit'], s['b0'], s['b1'])
        ax.plot(xs, pi_lo, ':', color=pi_c, linewidth=2, label='95% PI')
        ax.plot(xs, pi_hi, ':', color=pi_c, linewidth=2)

        text = (fr"$\beta_1$ = {s['b1']:.3f}, $\beta_0$ = {s['b0']:.3f}" "\n"
                fr"SE $\beta_1$ = {s['b1_std_err']:.3f}, SE $\beta_0$ = {s['b0_std_err']:.3f}" "\n"
                fr"CI $\beta_1$: ({s['ci_b1'][0]:.3f}, {s['ci_b1'][1]:.3f})" "\n"
                fr"CI $\beta_0$: ({s['ci_b0'][0]:.3f}, {s['ci_b0'][1]:.3f})" "\n"
                fr"PI $\beta_1$ = {s['b1']:.3f}, PI $\beta_0$: ({s['pi_b0'][0]:.3f}, {s['pi_b0'][1]:.3f})" "\n"
                fr"within PI: {100 * s['pi_cover']:.3f}%" "\n"
                + (fr"$R^2$ (pooled) = {s['r2']:.3f}" "\n" fr"$R^2$ (per-event mean) = {s['r2_event_mean']:.3f}"
                   if 'r2_event_mean' in s else fr"$R^2$ = {s['r2']:.3f}"))
        ax.text(0.03, 0.97, text, fontsize=fontsize, verticalalignment='top', transform=ax.transAxes,
                bbox=self._info_box(best))
        ax.set_title(title)
        ax.set_xlabel('predicted proton (mean of runs)')
        ax.set_ylabel('true proton')
        ax.legend(loc='lower right', fontsize=fontsize)

    def _event_legend(self, fig, events: list[int], labels: dict, title: str, fontsize: int, ncol: int = 1,
                      x: float = 0.8) -> None:
        # One shared legend per figure: marker + colour per event, with the given text per event.
        # Its left edge sits at x (figure fraction), right where the panels end.
        handles = [Line2D([], [], linestyle='', markersize=8, **self._event_style(e)) for e in events]
        fig.legend(handles, [labels[e] for e in events], loc='center left', bbox_to_anchor=(x, 0.5),
                   fontsize=fontsize, title=title, title_fontsize=fontsize, ncol=ncol, frameon=False)

    def _true_vs_pred_models(self, pred_data: tuple[dict]) -> list[str]:
        # Baselines first, then champions, in the order Analysis lists them; only models with data.
        present = {mt for d in pred_data for mt in d['data']}
        return [mt for mt in self.analysis.baseline_names + self.analysis.champion_names if mt in present]

    def _all_true_vs_pred_stats(self, pred_data: tuple[dict]) -> tuple[dict, dict]:
        # Stats for every (model, t, phases), plus {(t, phases): model with the highest R^2}.
        stats = {(mt, t, p): self._true_vs_pred_stats(pred_data, mt, t, p)
                 for mt in self._true_vs_pred_models(pred_data)
                 for t in self.analysis.times for p in self.analysis.phases}
        best_r2 = {}
        for (mt, t, p), s in stats.items():
            if s is None or np.isnan(s['r2']):
                continue
            if (t, p) not in best_r2 or s['r2'] > stats[(best_r2[(t, p)], t, p)]['r2']:
                best_r2[(t, p)] = mt
        return stats, best_r2

    def plot_true_vs_pred(self, pred_data: tuple[dict]) -> None:
        # One PNG per (model, phases): left t+6, right t+12.
        # Saved as results/true_vs_pred/bootstrap_mean_estimates_{model}_{phases}.png
        out_dir = self.analysis.results_folder / 'true_vs_pred'
        out_dir.mkdir(parents=True, exist_ok=True)
        times = self.analysis.times
        stats, best_r2 = self._all_true_vs_pred_stats(pred_data)
        for mt in self._true_vs_pred_models(pred_data):
            for p in self.analysis.phases:
                fig = Figure(figsize=(18, 7.5), dpi=100)
                axes = fig.subplots(1, len(times), squeeze=False)[0]
                fig.suptitle(f'{mt}, {p}: true vs predicted, event windows onset to peak (bootstrap mean estimates)')
                for ax, t in zip(axes, times):
                    s = stats[(mt, t, p)]
                    if s is None:
                        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(t)
                        continue
                    # green box: best R^2 of all models for this horizon and phases setting
                    self._draw_true_vs_pred(ax, s, title=t, best=best_r2.get((t, p)) == mt)

                # Legend: every event in either panel, with its R^2 at each horizon (- = not in that test split)
                per_t = {t: (stats[(mt, t, p)] or {}).get('event_r2', {}) for t in times}
                events = sorted(set().union(*per_t.values()))
                labels = {e: self._event_label(e) + ': ' + ' / '.join(
                              f"{per_t[t][e]:.3f}" if e in per_t[t] and np.isfinite(per_t[t][e]) else '-' for t in times)
                          for e in events}
                right = 0.66 if len(events) > 22 else 0.8      # two legend columns need more room
                self._event_legend(fig, events, labels, title=r'event: $R^2$ ' + ' / '.join(times),
                                   fontsize=9, ncol=2 if len(events) > 22 else 1, x=right + 0.005)
                fig.tight_layout(rect=(0, 0, right, 0.96))
                fig.savefig(out_dir / f'bootstrap_mean_estimates_{mt}_{p}.png', dpi=150)

    def plot_true_vs_pred_all_models(self, pred_data: tuple[dict]) -> None:
        # Experiment: every model in one figure per phases, rows = models, columns = t+6 / t+12.
        # Saved as results/true_vs_pred/all_models_{phases}.png
        out_dir = self.analysis.results_folder / 'true_vs_pred'
        out_dir.mkdir(parents=True, exist_ok=True)
        times = self.analysis.times
        models = self._true_vs_pred_models(pred_data)
        stats, best_r2 = self._all_true_vs_pred_stats(pred_data)
        for p in self.analysis.phases:
            fig = Figure(figsize=(17, 6 * len(models)), dpi=100)
            axes = fig.subplots(len(models), len(times), squeeze=False)
            fig.suptitle(f'All models, {p}: true vs predicted, event windows onset to peak (bootstrap mean estimates)',
                         y=1 - 0.15 / (6 * len(models)))   # pin to the top edge of this very tall figure
            for row, mt in enumerate(models):
                for col, t in enumerate(times):
                    ax = axes[row, col]
                    s = stats[(mt, t, p)]
                    if s is None:
                        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{mt}, {t}')
                        continue
                    # green box: best R^2 in this column
                    self._draw_true_vs_pred(ax, s, title=f'{mt}, {t}', fontsize=9, best=best_r2.get((t, p)) == mt)
            events = sorted({int(e) for s in stats.values() if s is not None for e in s['event_r2']})
            self._event_legend(fig, events, {e: self._event_label(e) for e in events}, title='event', fontsize=11,
                               x=0.875)
            fig.tight_layout(rect=(0, 0, 0.87, 1 - 0.6 / (6 * len(models))))   # right: event legend; top: suptitle
            fig.savefig(out_dir / f'all_models_{p}.png', dpi=100)

    # ---------- true vs predicted, one figure per event ----------

    def _event_true_vs_pred_stats(self, pred_data: tuple[dict], mt: str, t: str, p: str, e: str) -> dict | None:
        # One event for one (model, horizon, phases). The event appears in 1-5 datasets; the points are
        # each of those datasets' run-mean predictions vs the event's true protons, and every value in the
        # box is the bootstrap mean across those datasets and all runs. None if the model never saw it in test.
        record = next((d for d in pred_data if p in d['data'].get(mt, {}).get(t, {})), None)
        if record is None:
            return None
        datasets = [ds for ds, events in record['data'][mt][t][p].items() if e in events]
        if not datasets:
            return None

        def pooled_mean(name, idx=None):
            vals = [record[name][mt][t][p][ds][e] for ds in datasets]
            if idx is not None:
                vals = [v[idx] for v in vals]
            return float(np.mean(np.concatenate([np.ravel(v) for v in vals])))

        x = np.concatenate([record['data'][mt][t][p][ds][e].mean(axis=1) for ds in datasets])
        y = np.concatenate([self.analysis.true_protons[t][ds][e] for ds in datasets])

        s = {name: pooled_mean(name) for name in ('b0', 'b1', 'b1_std_err', 'b0_std_err', 'mean', 'ssx', 's2', 't_crit', 'r2')}
        s['ci_b1'] = (pooled_mean('ci', 0), pooled_mean('ci', 1))
        s['ci_b0'] = (pooled_mean('ci_b0', 0), pooled_mean('ci_b0', 1))
        half = s['t_crit'] * np.sqrt(s['s2'])
        s['pi_b0'] = (s['b0'] - half, s['b0'] + half)
        pi_lo, pi_hi = self.analysis.get_prediction_predictions_intervals(
            x, s['mean'], s['ssx'], s['s2'], s['t_crit'], s['b0'], s['b1'])
        s['pi_cover'] = float(np.mean((y >= pi_lo) & (y <= pi_hi)))
        s['x'], s['y'], s['n_datasets'] = x, y, len(datasets)
        return s

    def plot_true_vs_pred_events(self, pred_data: tuple[dict]) -> None:
        # One PNG per (event, phases): rows = models, columns = t+6 / t+12.
        # Saved as results/true_vs_pred/phases/event_22.png and results/true_vs_pred/no_phases/...
        # Green box = best R^2 of all models in that column. Events never in a test split are skipped.
        times = self.analysis.times
        models = self._true_vs_pred_models(pred_data)
        events = sorted({e for t in times for ds in self.analysis.event_slices[t] for e in self.analysis.event_slices[t][ds]},
                        key=lambda e: int(e[1:]))
        folders = {'phases': 'phases', 'nophases': 'no_phases'}

        for p in self.analysis.phases:
            out_dir = self.analysis.results_folder / 'true_vs_pred' / folders[p]
            out_dir.mkdir(parents=True, exist_ok=True)
            wins = {t: {mt: 0 for mt in models} for t in times}   # events each model predicted best, per horizon
            for e in events:
                stats = {(mt, t): self._event_true_vs_pred_stats(pred_data, mt, t, p, e) for mt in models for t in times}
                best_r2 = {}
                for (mt, t), st in stats.items():
                    if st is not None and np.isfinite(st['r2']) and (t not in best_r2 or st['r2'] > stats[(best_r2[t], t)]['r2']):
                        best_r2[t] = mt
                for t, mt in best_r2.items():
                    wins[t][mt] += 1

                fig = Figure(figsize=(14, 6 * len(models)), dpi=100)
                axes = fig.subplots(len(models), len(times), squeeze=False)
                fig.suptitle(f'{self._event_title(e)}\n{p}: true vs predicted, onset to peak',
                             y=1 - 0.15 / (6 * len(models)))
                for row, mt in enumerate(models):
                    for col, t in enumerate(times):
                        ax = axes[row, col]
                        st = stats[(mt, t)]
                        if st is None:
                            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{mt}, {t}')
                            continue
                        self._draw_true_vs_pred(ax, st, title=f"{mt}, {t} ({st['n_datasets']} datasets)",
                                                fontsize=9, best=best_r2.get(t) == mt)
                fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / (6 * len(models))))
                fig.savefig(out_dir / f'event_{int(e[1:]):02d}.png', dpi=100)   # e.g. event_07.png; date is in the title

            self._plot_best_model_counts(wins, p, len(events), out_dir / 'best_model_counts.png')

    def _plot_best_model_counts(self, wins: dict, phases: str, n_events: int, path) -> None:
        # Grouped bars: for each model, how many events it had the best R^2 on (the green box), t+6 vs t+12.
        times = list(wins)
        models = list(wins[times[0]])
        colors = {times[0]: self.palette[0], times[-1]: self.palette[5]}   # navy / orange, one per horizon
        width = 0.8 / len(times)

        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.subplots()
        x = np.arange(len(models))
        for i, t in enumerate(times):
            counts = [wins[t][mt] for mt in models]
            bars = ax.bar(x + (i - (len(times) - 1) / 2) * width, counts, width * 0.95, color=colors[t], label=t)
            ax.bar_label(bars, labels=[str(c) if c else '' for c in counts], padding=2, fontsize=9)
        ax.set_xticks(x, models)
        ax.set_ylabel('events with the best $R^2$')
        ax.set_title(f'{phases}: which model predicted each event best (highest $R^2$, {n_events} events)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=150)

    # ---------- per-event metric histograms ('events' bootstrap mode) ----------

    def _best_event_models(self, means: dict, metric: str) -> list[str]:
        # Every model tied for the best mean on one metric (ties are shared wins). Posner can't win a lag.
        rule = self.metric_best[metric]
        scored = {mt: v for mt, v in means.items()
                  if np.isfinite(v) and not (rule == 'zero' and mt in self.lag_disqualified)}
        if not scored:
            return []
        score = {mt: {'min': v, 'max': -v, 'zero': abs(v)}[rule] for mt, v in scored.items()}
        top = min(score.values())
        return [mt for mt, sc in score.items() if np.isclose(sc, top)]

    def _draw_event_histogram(self, ax, values: np.ndarray, mean: float, se: float, sb: float, best: bool) -> None:
        # Same look as the per-dataset histograms: bars, mean, mean +/- SE (shaded), 95% percentile CI, info box.
        bar_c, mean_c, se_c, sb_c, ci_c = self.palette[:5]
        if len(values) == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            return
        if np.unique(values).size <= 1:
            ax.text(0.5, 0.5, f'all values = {values[0]:.3f}\n(n={len(values)})', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, bbox=self._info_box(best))
            ax.set_xticks([]); ax.set_yticks([])
            return

        n_bins = max(4, len(np.histogram_bin_edges(values, bins='auto')) - 1)
        ax.hist(values, bins=n_bins, color=bar_c, edgecolor='white')
        ci_lo, ci_hi = np.percentile(values, [2.5, 97.5])
        top = ax.get_ylim()[1]
        if np.isfinite(se):
            ax.axvspan(mean - se, mean + se, color=se_c, alpha=0.15, zorder=0)
            ax.vlines([mean - se, mean + se], 0, top, colors=se_c, linestyles='--', linewidth=1.5, label='mean ± SE')
        ax.vlines(mean, 0, top, colors=mean_c, linestyles='--', linewidth=1.5, label='mean')
        if np.isfinite(sb):
            ax.vlines(mean + sb, 0, top, colors=sb_c, linewidth=1.5, label='mean + SB')
        ax.vlines([ci_lo, ci_hi], 0, top, colors=ci_c, linestyles=':', linewidth=2, label='95% CI (percentile)')
        ax.set_ylim(0, top)
        text = (f"n = {len(values)}\nmean = {mean:.3f}\nSE = {se:.3f}\nSB = {sb:.3f}\n"
                f"95% CI: ({ci_lo:.3f}, {ci_hi:.3f})")
        ax.text(0.97, 0.95, text, fontsize=8, ha='right', va='top', transform=ax.transAxes, bbox=self._info_box(best))

    def plot_event_metric_histograms(self, mean_estimate_data: tuple[pd.DataFrame]) -> None:
        # Needs Analysis in 'events' bootstrap mode. One PNG per (phases, metric, event): rows = models,
        # columns = t+6 / t+12, saved as metrics_analysis/hists/<phases>/<metric>/event_NN.png, plus a
        # best_model_counts.png per metric folder. Values = mean_estimate_data (one row per dataset, seed and
        # event); mean and SE come from the mean_bootstrap_estimates / bootstrap_standard_errors CSVs.
        a = self.analysis
        metrics, times = a.desired_results_metrics, a.times
        values = pd.concat(mean_estimate_data, ignore_index=True)
        means = pd.concat([self._load_summary(prefix, 'mean_bootstrap_estimates') for prefix in ('baseline', 'champion')])
        ses = pd.concat([self._load_summary(prefix, 'bootstrap_standard_errors') for prefix in ('baseline', 'champion')])
        sbs = pd.concat([self._load_summary(prefix, 'bootstrap_standard_bias') for prefix in ('baseline', 'champion')])
        groups = {k: g for k, g in values.groupby(a.bootstrap_keys)}
        models = [mt for mt in a.baseline_names + a.champion_names if mt in set(values['model_type'])]
        events = sorted(values['event'].unique(), key=lambda e: int(e[1:]))

        for p in a.phases:
            for m in metrics:
                out_dir = a.results_folder / a.hists_folder / self.phase_folders[p] / self.metric_folders[m]
                out_dir.mkdir(parents=True, exist_ok=True)
                wins = {t: {mt: 0 for mt in models} for t in times}
                for e in events:
                    best = {}
                    for t in times:
                        event_means = {mt: means.loc[(mt, p, t, e), m] for mt in models if (mt, p, t, e) in means.index}
                        best[t] = self._best_event_models(event_means, m)
                        for mt in best[t]:
                            wins[t][mt] += 1          # ties: every tied model gets the win

                    fig = Figure(figsize=(12, 2.8 * len(models)), dpi=100)
                    axes = fig.subplots(len(models), len(times), squeeze=False)
                    fig.suptitle(f'{self._event_title(e)}\n{p}: {m} per event')
                    for row, mt in enumerate(models):
                        for col, t in enumerate(times):
                            ax = axes[row, col]
                            ax.set_title(f'{mt}, {t}', fontsize=10)
                            key = (mt, p, t, e)
                            v = groups[key][m].dropna().to_numpy() if key in groups else np.array([])
                            mean = means.loc[key, m] if key in means.index else np.nan
                            se = ses.loc[key, m] if key in ses.index else np.nan
                            sb = sbs.loc[key, m] if key in sbs.index else np.nan
                            self._draw_event_histogram(ax, v, mean, se, sb, best=mt in best[t])
                    handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat
                                            if ax.get_legend_handles_labels()[0]), ([], []))
                    fig.legend(handles, labels, loc='upper right', ncol=3, fontsize=9)
                    fig.tight_layout(rect=(0, 0, 1, 0.97))
                    fig.savefig(out_dir / f'event_{int(e[1:]):02d}.png', dpi=80)

                self._plot_event_best_counts(wins, p, m, len(events), out_dir / 'best_model_counts.png')

    def plot_train_time(self, train_times: pd.DataFrame) -> None:
        # One PNG per phases: rows = models, columns = t+6 / t+12, saved as metrics_analysis/hists/<phases>/training_time.png.
        # Values = one train time per dataset (seeds averaged, like the per-dataset histograms); the info box has
        # mean, SE (1/(B-1)), SB (1/B) and the 95% percentile CI. Green box = fastest model in that column.
        a = self.analysis
        per_dataset = (train_times.groupby(['model_type', 'phases', 't', 'dataset'])['Train Time'].mean()
                       .reset_index())
        models = [mt for mt in a.baseline_names + a.champion_names if mt in set(per_dataset['model_type'])]
        groups = {k: g['Train Time'].dropna().to_numpy() for k, g in per_dataset.groupby(['model_type', 'phases', 't'])}

        for p in a.phases:
            out_dir = a.results_folder / a.hists_folder / self.phase_folders[p]
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = Figure(figsize=(12, 2.8 * len(models)), dpi=100)
            axes = fig.subplots(len(models), len(a.times), squeeze=False)
            fig.suptitle(f'{p}: train time (s), one value per dataset (seeds averaged)')
            for col, t in enumerate(a.times):
                col_means = {mt: groups[(mt, p, t)].mean() for mt in models if len(groups.get((mt, p, t), [])) > 0}
                fastest = min(col_means, key=col_means.get) if col_means else None
                for row, mt in enumerate(models):
                    ax = axes[row, col]
                    v = groups.get((mt, p, t), np.array([]))
                    ax.set_title(f'{mt}, {t}', fontsize=10)
                    if len(v):
                        diffs = v - v.mean()
                        se = np.sqrt((diffs ** 2).sum() / (len(v) - 1)) if len(v) > 1 else np.nan   # get_standard_error
                        sb = np.sqrt((diffs ** 2).sum() / len(v))                                   # get_standard_bias
                        self._draw_event_histogram(ax, v, v.mean(), se, sb, best=mt == fastest)
                    else:
                        self._draw_event_histogram(ax, v, np.nan, np.nan, np.nan, best=False)
                    ax.set_xlabel('seconds', fontsize=8)
            handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat
                                    if ax.get_legend_handles_labels()[0]), ([], []))
            fig.legend(handles, labels, loc='upper right', ncol=4, fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(out_dir / 'training_time.png', dpi=120)

    def _plot_event_best_counts(self, wins: dict, phases: str, metric: str, n_events: int, path) -> None:
        # Bars = events each model was best on for one metric, t+6 vs t+12. Posner is left out of the lags.
        times = list(wins)
        rule = self.metric_best[metric]
        models = [mt for mt in wins[times[0]] if not (rule == 'zero' and mt in self.lag_disqualified)]
        colors = {times[0]: self.palette[0], times[-1]: self.palette[5]}
        width = 0.8 / len(times)

        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.subplots()
        x = np.arange(len(models))
        for i, t in enumerate(times):
            counts = [wins[t][mt] for mt in models]
            bars = ax.bar(x + (i - (len(times) - 1) / 2) * width, counts, width * 0.95, color=colors[t], label=t)
            ax.bar_label(bars, labels=[str(c) if c else '' for c in counts], padding=2, fontsize=9)
        rule_text = {'min': 'lowest', 'max': 'highest', 'zero': 'closest to 0'}[rule]
        ax.set_title(f'{phases}: which model was best on each event, {metric} ({rule_text} mean, {n_events} events)')
        ax.set_xticks(x, models)
        ax.set_ylabel('events where the model was best')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc='upper left')
        if rule == 'zero':
            ax.text(0.99, 0.98, self.lag_note, transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=self._info_box(False))
        ax.text(0.0, -0.12, 'ties count as a win for every tied model', transform=ax.transAxes,
                fontsize=8, color='0.4', va='top')
        fig.tight_layout()
        fig.savefig(path, dpi=120)

    def plot_hypothesis_grid(self, test_table: pd.DataFrame, test_folder: str, title_test: str,
                             hypothesis_lines: list[str], reject_label: str) -> None:
        # One PNG per champion: a reject / don't-reject grid against each baseline, for one hypothesis test table
        # (Analysis.perform_welch_test or perform_levene_test). Rows = metrics, columns = t+6 / t+12 x phases /
        # no phases, each cell shows the one-sided p-value. Green = reject H0 (reject_label), red = don't reject,
        # grey = not enough data. Saved as results/metrics_analysis/hypothesis_grids/<test_folder>/<champion>.png
        from matplotlib.patches import Rectangle, FancyBboxPatch
        analysis = self.analysis
        state_colours = {'reject': '#0ca30c', 'keep': '#d03b3b', 'na': '#e1e0d9'}
        state_words = {'reject': 'Reject', 'keep': "Don't reject", 'na': 'n/a'}
        metrics = [metric for metric in analysis.results_metrics if metric in set(test_table['metric'])]
        columns = [(horizon, phases) for horizon in analysis.times for phases in analysis.phases]
        column_labels = [f"{horizon}\n{'phases' if phases == 'phases' else 'no phases'}" for horizon, phases in columns]
        baseline_names = [name for name in analysis.baseline_names if name in set(test_table['baseline'])]
        p_values = test_table.set_index(['model_type', 'baseline', 'metric', 't', 'phases'])['p_value']
        significance_level = analysis.significance_level

        output_folder = analysis.results_folder / 'metrics_analysis' / 'hypothesis_grids' / test_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        for champion_name in [name for name in analysis.champion_names if name in set(test_table['model_type'])]:
            figure = Figure(figsize=(12, 3.0 * len(baseline_names) + 1.0), dpi=100)
            figure.suptitle(f'{champion_name} vs baselines: {title_test}, alpha = {significance_level}', fontsize=13)
            layout = figure.add_gridspec(len(baseline_names), 2, width_ratios=[1, 2.4], wspace=0.25, hspace=0.45)

            # Left: the hypotheses and the colour legend
            hypothesis_panel = figure.add_subplot(layout[:, 0])
            hypothesis_panel.axis('off')
            hypothesis_panel.text(0.02, 0.5, '\n'.join(hypothesis_lines), va='center', ha='left', fontsize=9,
                                  family='monospace', bbox=dict(boxstyle='round,pad=0.8', fc='#f7f7f5', ec='#c3c2b7'))
            legend_patches = [Rectangle((0, 0), 1, 1, fc=state_colours[state]) for state in ('reject', 'keep', 'na')]
            hypothesis_panel.legend(legend_patches, [f'Reject H0 ({reject_label})', "Don't reject H0", 'Not enough data'],
                                    loc='lower left', fontsize=9, frameon=False)

            # Right: one grid per baseline
            for baseline_index, baseline_name in enumerate(baseline_names):
                grid_axes = figure.add_subplot(layout[baseline_index, 1])
                grid_axes.set_title(f'vs {baseline_name}', fontsize=11)
                for row_index, metric in enumerate(metrics):
                    for column_index, (horizon, phases) in enumerate(columns):
                        p_value = p_values.get((champion_name, baseline_name, metric, horizon, phases), np.nan)
                        state = 'na' if pd.isna(p_value) else 'reject' if p_value < significance_level else 'keep'
                        grid_axes.add_patch(FancyBboxPatch((column_index + 0.04, row_index + 0.06), 0.92, 0.88,
                                                           boxstyle='round,pad=0,rounding_size=0.06',
                                                           fc=state_colours[state], ec='none'))
                        cell_text = state_words[state] if state == 'na' else f'{state_words[state]}\np = {p_value:.3g}'
                        grid_axes.text(column_index + 0.5, row_index + 0.5, cell_text, ha='center', va='center',
                                       fontsize=8, color='#0b0b0b' if state == 'na' else 'white')
                grid_axes.set_xlim(0, len(columns))
                grid_axes.set_ylim(len(metrics), 0)
                grid_axes.set_xticks(np.arange(len(columns)) + 0.5, column_labels, fontsize=9)
                grid_axes.set_yticks(np.arange(len(metrics)) + 0.5, metrics, fontsize=9)
                grid_axes.xaxis.tick_top()
                grid_axes.tick_params(length=0)
                for spine in grid_axes.spines.values():
                    spine.set_visible(False)

            figure.savefig(output_folder / f'{champion_name}.png', dpi=120, bbox_inches='tight')

    def plot_welch_grids(self, welch_test: pd.DataFrame) -> None:
        # Is the champion's mean better? -> results/metrics_analysis/hypothesis_grids/welch/<champion>.png
        alternative_by_metric = welch_test.drop_duplicates('metric').set_index('metric')['h_a']
        metrics = [metric for metric in self.analysis.results_metrics if metric in alternative_by_metric.index]
        metric_width = max(len(metric) for metric in metrics)
        shorten = lambda hypothesis: hypothesis.replace('champion', 'c').replace('baseline', 'b')
        hypothesis_lines = ['H0: champion no better', 'HA: (reject => champion better)', '']
        hypothesis_lines += [f'{metric:<{metric_width}}  {shorten(alternative_by_metric[metric])}' for metric in metrics]
        hypothesis_lines += ['', 'c = champion, b = baseline', 'lags: distance from 0', 'p = one-sided p-value']
        self.plot_hypothesis_grid(welch_test, 'welch', 'one-sided Welch t-test on the per-dataset means',
                                  hypothesis_lines, 'champion better')

    def plot_levene_grids(self, levene_test: pd.DataFrame) -> None:
        # Is the champion's spread smaller? -> results/metrics_analysis/hypothesis_grids/levene/<champion>.png
        hypothesis_lines = ['H0: var(c) >= var(b)', 'HA: var(c) <  var(b)', '(reject => champion has',
                            ' a smaller spread)', '',
                            'every metric, raw values', '(lags not converted to |lag|)', '',
                            'c = champion, b = baseline', 'p = one-sided p-value']
        self.plot_hypothesis_grid(levene_test, 'levene', "one-sided Levene's test (median centre) on the per-dataset values",
                                  hypothesis_lines, 'smaller spread')

    # ---------- data overview: the input series and the bootstrap datasets -> photos/ ----------

    def _flux_channels(self) -> dict:
        # column -> (label, colour), the same in every overview plot
        return {'electron':      ('Electron (low energy)',  self.palette[1]),
                'electron_high': ('Electron (high energy)', self.palette[3]),
                'proton':        ('Proton',                 self.palette[7])}

    def _event_rows(self, data_folder: str) -> list[tuple[int, int]]:
        # Row span of each event in data_0: 36 steps (3 h) before onset to the event end, as the bootstrap keeps them
        with open(f'{data_folder}/event_indices.txt') as f:
            return [(max(int(line.split()[0]) - 36, 0), int(line.split()[3])) for line in f if line.strip()]

    def _flux_limits(self, df: pd.DataFrame) -> dict:
        # Fixed y-range per channel (from data_0), so the original and every bootstrap share the same scale
        limits = {}
        for col in self._flux_channels():
            low, high = df[col].min(), df[col].max()
            pad = 0.04 * (high - low)
            limits[col] = (low - pad, high + pad)
        return limits

    def _draw_flux_panels(self, fig, x, df: pd.DataFrame, spans: list, title: str, xlabel: str,
                          limits: dict) -> None:
        # One panel per channel (shared x), events shaded, ln(10) = 10 pfu threshold on the proton panel
        channels = self._flux_channels()
        axes = fig.subplots(len(channels), 1, sharex=True)
        fig.suptitle(title, fontsize=13)
        for ax, (col, (label, colour)) in zip(axes, channels.items()):
            for start, end in spans:
                ax.axvspan(start, end, color=self.sep_c, alpha=0.45, lw=0)
            ax.plot(x, df[col].to_numpy(), color=colour, lw=0.4, rasterized=True)
            ax.set_ylim(*limits[col])
            ax.set_ylabel(f'{label}\nln(flux)', fontsize=9)
            ax.grid(axis='y', color='#e1e0d9', lw=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        axes[-1].axhline(np.log(10), color='#0b0b0b', ls='--', lw=0.8, label='ln(10) = 10 pfu')
        handles = [Line2D([], [], color=self.sep_c, alpha=0.45, lw=8, label='SEP events'),
                   Line2D([], [], color='#0b0b0b', ls='--', lw=0.8, label='ln(10) = 10 pfu')]
        axes[0].legend(handles=handles, loc='upper right', fontsize=8, frameon=False)
        axes[-1].set_xlabel(xlabel)
        fig.tight_layout()

    def plot_flux_series(self, data_folder: str = 'data', out_dir: str = 'photos') -> None:
        # The whole original series (data_0): electron low, electron high and proton, events shaded.
        # Saved as photos/flux_series.png
        df = pd.read_csv(f'{data_folder}/data_0.csv', usecols=['time', 'electron', 'electron_high', 'proton'])
        time = pd.to_datetime(df['time'])
        spans = [(time.iloc[s], time.iloc[min(e, len(df) - 1)]) for s, e in self._event_rows(data_folder)]
        limits = self._flux_limits(df)
        # Missing data: start a new line segment after any gap over 1 h instead of drawing straight across it
        gap = time.diff() > pd.Timedelta(hours=1)
        df = df.mask(gap, np.nan)
        fig = Figure(figsize=(16, 8))
        self._draw_flux_panels(fig, time, df, spans, f'Particle flux, {time.iloc[0]:%Y} to {time.iloc[-1]:%Y} '
                               f'({len(spans)} SEP events shaded)', 'Date', limits)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / 'flux_series.png', dpi=150)

    def plot_events(self, data_folder: str = 'data', out_dir: str = 'photos') -> None:
        # Every SEP event in its own PNG: 3 h before onset to the event end, the three channels, and the
        # onset / threshold / peak times marked and labelled. Saved as photos/events/event_e<NN>.png
        import matplotlib.dates as mdates
        df = pd.read_csv(f'{data_folder}/data_0.csv', usecols=['time', 'electron', 'electron_high', 'proton'])
        timestamps = read_event_timestamps(f'{data_folder}/event_timestamps.txt')
        events_dir = Path(out_dir) / 'events'
        events_dir.mkdir(parents=True, exist_ok=True)

        for n, (start, end) in enumerate(self._event_rows(data_folder)):
            window = df.iloc[start:end + 1]
            time = pd.to_datetime(window['time'])
            onset, threshold, peak, event_end = (pd.to_datetime(t) for t in timestamps[n])

            fig = Figure(figsize=(12, 5.5))
            ax = fig.subplots()
            for col, (label, colour) in self._flux_channels().items():
                ax.plot(time, window[col], color=colour, lw=2, label=label)
            ax.axhline(np.log(10), color='#0b0b0b', ls='--', lw=0.8, label='ln(10) = 10 pfu')

            # Marker lines, labelled with their time; staggered because onset and threshold are often close,
            # and right-aligned near the right edge so the label stays inside the plot.
            # Headroom above the data keeps the labels off the lines.
            y_low, y_high = ax.get_ylim()
            ax.set_ylim(y_low, y_high + 0.3 * (y_high - y_low))
            span = (time.iloc[-1] - time.iloc[0]).total_seconds()
            for k, (t, name) in enumerate(((onset, 'onset'), (threshold, 'threshold'), (peak, 'peak'))):
                ax.axvline(t, color='#898781', lw=0.8, ls=':')
                right = (t - time.iloc[0]).total_seconds() > 0.75 * span
                ax.annotate(f'{name} {t:%Y-%m-%d %H:%M}', (t, 1), xycoords=('data', 'axes fraction'),
                            xytext=(-3 if right else 3, -12 - 13 * k), textcoords='offset points',
                            ha='right' if right else 'left', fontsize=9, color='#5f5e5a',
                            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))

            ax.set_title(f'Event e{n + 1}: onset {onset:%Y-%m-%d %H:%M}, end {event_end:%Y-%m-%d %H:%M} UTC',
                         fontsize=12)
            ax.set_ylabel('ln(flux)')
            ax.set_xlabel('Time (UTC)')
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.grid(axis='y', color='#e1e0d9', lw=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9, frameon=False)
            fig.tight_layout()
            fig.savefig(events_dir / f'event_e{n + 1:02d}.png', dpi=150)

    def plot_bootstrap_series(self, data_folder: str = 'data', out_dir: str = 'photos',
                              bootstraps: tuple[int, ...] | None = None) -> None:
        # Bootstrapped series, same layout and y-scale as flux_series.png. The bootstrap shuffles ~7 h background
        # blocks and whole events, so its own timestamps are out of order. Row i is plotted at
        # data_0's start date + 5 min x i instead: a continuous calendar in the order the models see the rows.
        # (data_0's own dates have missing-data gaps; the bootstrap has none, so it isn't put on those.)
        # Shading shows where each event landed. bootstraps=None plots every data_<k>.csv (k >= 1).
        # Saved as photos/bootstraps/bootstrap_<k>.png
        original_df = pd.read_csv(f'{data_folder}/data_0.csv', usecols=['time', 'electron', 'electron_high', 'proton'])
        original = original_df['time']
        limits = self._flux_limits(original_df)
        dates = pd.Series(pd.to_datetime(original.iloc[0]) + pd.Timedelta(minutes=5) * np.arange(len(original)))
        event_times = set()
        for s, e in self._event_rows(data_folder):
            event_times.update(original.iloc[s:e])

        if bootstraps is None:
            bootstraps = sorted(int(f.stem.split('_')[1]) for f in Path(data_folder).glob('data_*.csv')
                                if f.stem.split('_')[1] != '0')
        boot_dir = Path(out_dir) / 'bootstraps'
        boot_dir.mkdir(parents=True, exist_ok=True)
        for k in bootstraps:
            df = pd.read_csv(f'{data_folder}/data_{k}.csv', usecols=['time', 'electron', 'electron_high', 'proton'])
            spans = self._bootstrap_spans(df, event_times, dates)
            fig = Figure(figsize=(16, 8))
            self._draw_flux_panels(fig, dates, df, spans, f'Bootstrap dataset data_{k}: block-shuffled background '
                                   f'and events (events shaded)', 'Date (data_0 start + 5 min per row)', limits)
            fig.savefig(boot_dir / f'bootstrap_{k}.png', dpi=150)

    @staticmethod
    def _bootstrap_spans(df: pd.DataFrame, event_times: set, dates: pd.Series) -> list:
        # (start, end) dates of each run of rows in a bootstrap dataset that came from an event
        in_event = df['time'].isin(event_times).to_numpy().astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], in_event, [0]])))
        return [(dates.iloc[a], dates.iloc[min(b, len(df) - 1)]) for a, b in zip(edges[::2], edges[1::2])]

    @matplotlib.rc_context(PAPER_RC)
    def plot_bootstrap_stack(self, data_folder: str = 'data', bootstraps: tuple[int, ...] = (1,),
                             out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\evaluation'
                                            r'\time_intensity_profile\learning_and_evaluation_procedures\equations'
                             ) -> None:
        # The original series (data_0) on top, the given bootstrap datasets underneath: one panel each with
        # electron low, electron high and proton overlaid, same y-scale, events shaded, ln(10) line.
        # Every row is plotted at data_0's start + 5 min x row (as in plot_bootstrap_series) so the
        # panels line up row for row. Saved as <out_dir>/bootstrap_stack.png
        channels = self._flux_channels()
        cols = ['time', *channels]
        original = pd.read_csv(f'{data_folder}/data_0.csv', usecols=cols)
        dates = pd.Series(pd.to_datetime(original['time'].iloc[0]) + pd.Timedelta(minutes=5) * np.arange(len(original)))
        event_times = set()
        for s, e in self._event_rows(data_folder):
            event_times.update(original['time'].iloc[s:e])
        limits = self._flux_limits(original)

        datasets = [('Original', original)] + \
                   [(f'Bootstrap {k}', pd.read_csv(f'{data_folder}/data_{k}.csv', usecols=cols))
                    for k in bootstraps]
        y_range = (min(low for low, _ in limits.values()), max(high for _, high in limits.values()))
        fig = Figure(figsize=(12, 3 * len(datasets) + 0.6))
        axes = fig.subplots(len(datasets), 1, sharex=True)
        for ax, (name, df) in zip(axes, datasets):
            for start, end in self._bootstrap_spans(df, event_times, dates):
                ax.axvspan(start, end, color=self.sep_c, alpha=0.45, lw=0)
            for col, (label, colour) in channels.items():   # proton last, so it is drawn on top
                ax.plot(dates.iloc[:len(df)], df[col].to_numpy(), color=colour, lw=0.4, rasterized=True)
            ax.axhline(np.log(10), color='#0b0b0b', ls='--', lw=0.8)
            ax.set_ylim(*y_range)
            ax.set_ylabel('ln(flux)')
            ax.set_title(name, loc='left')
            ax.grid(axis='y', color='#e1e0d9', lw=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        handles = [Line2D([], [], color=colour, lw=1.5, label=label) for label, colour in channels.values()] + \
                  [Line2D([], [], color=self.sep_c, alpha=0.45, lw=8, label='SEP events'),
                   Line2D([], [], color='#0b0b0b', ls='--', lw=0.8, label='ln(10) = 10 pfu')]
        fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False)
        axes[-1].set_xlabel('Date (data_0 start + 5 min per row)')
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / 'bootstrap_stack.png', dpi=150)

    # ---------- MAE / MSE comparison: box plots of the per-dataset values, bars of the bootstrap means ----------

    def _error_metric_axes_grid(self, figure, title: str):
        # Rows = MAE, MSE; columns = t+6, t+12 (the two error metrics, lower is better)
        error_metrics = ['MAE', 'MSE']
        axes_grid = figure.subplots(len(error_metrics), len(self.analysis.times), squeeze=False)
        figure.suptitle(title, fontsize=13)
        return error_metrics, axes_grid

    def _model_order_and_colours(self, model_names_present: set) -> tuple[list[str], list[str]]:
        # Baselines first, then champions; baselines grey, champions blue (one colour per group, not per model)
        analysis = self.analysis
        ordered_models = [name for name in analysis.baseline_names + analysis.champion_names if name in model_names_present]
        group_colours = [self.baseline_c if name in analysis.baseline_names else self.champion_c for name in ordered_models]
        return ordered_models, group_colours

    def _style_error_axes(self, axes, ordered_models: list[str], metric: str, horizon: str) -> None:
        axes.set_title(f'{metric}, {horizon}', fontsize=11)
        axes.set_xticks(np.arange(len(ordered_models)), ordered_models, rotation=35, ha='right', fontsize=9)
        unit = 'ln flux squared' if metric == 'MSE' else 'ln flux'
        axes.set_ylabel(f'{metric} ({unit})', fontsize=9)
        axes.grid(axis='y', color='#e1e0d9', lw=0.5)
        axes.set_axisbelow(True)
        for side in ('top', 'right'):
            axes.spines[side].set_visible(False)

    def _error_group_legend(self, figure) -> None:
        group_patches = [Line2D([], [], color=self.baseline_c, lw=8, label='Baselines'),
                         Line2D([], [], color=self.champion_c, lw=8, label='Champions')]
        figure.legend(handles=group_patches, loc='upper right', ncol=2, fontsize=9, frameon=False)

    def plot_mae_mse(self, mean_estimates: tuple[pd.DataFrame], bootstrap_means: tuple[pd.DataFrame],
                     bootstrap_std_errors: tuple[pd.DataFrame]) -> None:
        # Two PNGs per phases, in results/metrics_analysis/hists/<phases>/:
        #   mae_mse_boxplot.png  box plot of each model's per-dataset values (mean_estimates), points on top
        #   mae_mse_bars.png     bar = mean of the bootstrap estimates (get_mean_bootstrap_estimates),
        #                        error bar = bootstrap standard error (get_bootstrap_standard_error)
        analysis = self.analysis
        per_dataset_values = pd.concat(mean_estimates, ignore_index=True)
        means_table = pd.concat(bootstrap_means, ignore_index=True).set_index(['model_type', 'phases', 't'])
        std_error_table = pd.concat(bootstrap_std_errors, ignore_index=True).set_index(['model_type', 'phases', 't'])
        ordered_models, group_colours = self._model_order_and_colours(set(per_dataset_values['model_type']))
        jitter = np.random.default_rng(0)

        for phases in analysis.phases:
            output_folder = analysis.results_folder / analysis.hists_folder / self.phase_folders[phases]
            output_folder.mkdir(parents=True, exist_ok=True)
            phase_values = per_dataset_values[per_dataset_values['phases'] == phases]
            n_datasets = phase_values.groupby(['model_type', 't']).size().max()

            # Box plot of the per-dataset values
            box_figure = Figure(figsize=(14, 9))
            error_metrics, axes_grid = self._error_metric_axes_grid(
                box_figure, f'{phases}: MAE and MSE per dataset (each box = {n_datasets} per-dataset values, seeds averaged)')
            for row_index, metric in enumerate(error_metrics):
                for column_index, horizon in enumerate(analysis.times):
                    axes = axes_grid[row_index, column_index]
                    horizon_values = phase_values[phase_values['t'] == horizon]
                    model_values = [horizon_values.loc[horizon_values['model_type'] == name, metric].dropna().to_numpy()
                                    for name in ordered_models]
                    boxes = axes.boxplot(model_values, positions=np.arange(len(ordered_models)), widths=0.55,
                                         patch_artist=True, showfliers=False,
                                         medianprops=dict(color='#0b0b0b', lw=1.5))
                    for box, colour in zip(boxes['boxes'], group_colours):
                        box.set(facecolor=to_rgba(colour, 0.35), edgecolor=colour, lw=1.2)
                    for whisker_or_cap in boxes['whiskers'] + boxes['caps']:
                        whisker_or_cap.set(color='#898781', lw=1)
                    for position, (values, colour) in enumerate(zip(model_values, group_colours)):
                        axes.scatter(position + jitter.uniform(-0.15, 0.15, len(values)), values, s=12,
                                     color=colour, edgecolor='white', lw=0.4, zorder=3)
                    self._style_error_axes(axes, ordered_models, metric, horizon)
            self._error_group_legend(box_figure)
            box_figure.tight_layout(rect=(0, 0, 1, 0.96))
            box_figure.savefig(output_folder / 'mae_mse_boxplot.png', dpi=130)

            # Bars of the bootstrap means with bootstrap standard error bars
            bar_figure = Figure(figsize=(14, 9))
            error_metrics, axes_grid = self._error_metric_axes_grid(
                bar_figure, f'{phases}: mean of the bootstrap estimates, error bars = bootstrap standard error')
            for row_index, metric in enumerate(error_metrics):
                for column_index, horizon in enumerate(analysis.times):
                    axes = axes_grid[row_index, column_index]
                    keys = [(name, phases, horizon) for name in ordered_models]
                    bar_heights = np.array([means_table[metric].get(key, np.nan) for key in keys], dtype=float)
                    bar_errors = np.array([std_error_table[metric].get(key, np.nan) for key in keys], dtype=float)
                    axes.bar(np.arange(len(ordered_models)), bar_heights, width=0.6, color=group_colours,
                             edgecolor='white', lw=2)
                    axes.errorbar(np.arange(len(ordered_models)), bar_heights, yerr=bar_errors, fmt='none',
                                  ecolor='#0b0b0b', elinewidth=1.2, capsize=4)
                    for position, (height, error) in enumerate(zip(bar_heights, bar_errors)):
                        if not np.isnan(height):
                            label_height = height + (0 if np.isnan(error) else error)
                            axes.annotate(f'{height:.2f}', (position, label_height), xytext=(0, 3),
                                          textcoords='offset points', ha='center', fontsize=7.5, color='#5f5e5a')
                    self._style_error_axes(axes, ordered_models, metric, horizon)
            self._error_group_legend(bar_figure)
            bar_figure.tight_layout(rect=(0, 0, 1, 0.96))
            bar_figure.savefig(output_folder / 'mae_mse_bars.png', dpi=130)

    @matplotlib.rc_context(PAPER_RC)
    def plot_mae_mse_grouped(self, mean_estimates: tuple[pd.DataFrame], bootstrap_means: tuple[pd.DataFrame],
                             bootstrap_std_errors: tuple[pd.DataFrame],
                             out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\bootstrap',
                             metrics: tuple[str, ...] = ('MAE', 'MSE'), kinds: tuple[str, ...] = ('box', 'bars'),
                             log_x: bool = False, exclude: frozenset = frozenset()) -> None:
        # Paper version of plot_mae_mse: horizontal, one figure per metric, and for each model four skinny
        # boxes / bars, one per (t+6, phases), (t+6, no phases), (t+12, phases), (t+12, no phases).
        # Works for any metric, e.g. metrics=('O2P lag',); for lags, green = closest to zero (posner included).
        # kinds picks the box plot and/or bar chart, log_x puts the values on a log axis, and the models in
        # exclude never get a green mark. (plot_train_time_grouped uses this for the training time.)
        # Saved as <out_dir>/<metric>_boxplot_grouped.png and <metric>_bars_grouped.png (e.g. o2p_lag_...)
        analysis = self.analysis
        values = pd.concat(mean_estimates, ignore_index=True)
        means_table = pd.concat(bootstrap_means, ignore_index=True).set_index(['model_type', 'phases', 't'])
        std_error_table = pd.concat(bootstrap_std_errors, ignore_index=True).set_index(['model_type', 'phases', 't'])
        models = [name for name in analysis.baseline_names + analysis.champion_names if name in set(values['model_type'])]
        n_baselines = sum(name in analysis.baseline_names for name in models)

        # (t, phases) combinations top to bottom within each model, with their palette colour and offset
        conditions = [(t, p) for t in analysis.times for p in analysis.phases]
        colours = dict(zip(conditions, [self.palette[1], self.palette[0], self.palette[5], self.palette[7]]))
        labels = {(t, p): f"${t}$, {'phases' if p == 'phases' else 'no phases'}" for t, p in conditions}
        height = 0.18
        offsets = dict(zip(conditions, (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * height))
        jitter = np.random.default_rng(0)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def style(ax, metric: str, best_label: str) -> None:
            ax.set_yticks(np.arange(len(models)), models)
            ax.set_ylim(len(models) - 0.5, -0.5)   # first model at the top
            ax.axhline(n_baselines - 0.5, color='#0b0b0b', ls='--', lw=0.8)   # baselines above, champions below
            unit = {'MAE': 'ln flux', 'MSE': 'ln flux squared',
                    'Train Time': 'seconds'}.get(metric, '5-min steps' if 'lag' in metric else '')
            ax.set_xlabel(f'{metric} ({unit})' if unit else metric)
            if log_x:
                # Start the log axis just below the smallest mean of the non-excluded models, so a ~0 value
                # (e.g. persistence's train time) doesn't stretch it; such models get a '≈ 0' label instead
                ax.set_xscale('log')
                model_means = means_table[metric].groupby(level='model_type').max()
                floor = 0.5 * model_means.drop(index=list(exclude), errors='ignore').min()
                ax.set_xlim(left=floor)
                for row, name in enumerate(models):
                    if model_means.get(name, np.nan) < floor:
                        ax.text(floor * 1.15, row, '≈ 0', va='center')
            ax.grid(axis='x', color='#e1e0d9', lw=0.5)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            handles = [Line2D([], [], color=colours[c], lw=8, label=labels[c]) for c in conditions]
            handles.append(Line2D([], [], color=self.best_c, lw=2.5, label=best_label))
            ax.figure.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
            ax.figure.tight_layout(rect=(0, 0, 1, 0.87))

        def lowest(numbers, rule: str = 'min', exclude: set = frozenset()) -> int | None:
            # index of the best value by rule ('min' smallest, 'zero' closest to zero, 'max' largest), ignoring
            # NaN and the excluded models; None if every value is NaN
            numbers = np.asarray(numbers, dtype=float)
            if rule == 'zero':
                numbers = np.abs(numbers)
            elif rule == 'max':
                numbers = -numbers
            numbers[[name in exclude or name in always_excluded for name in models]] = np.nan
            return None if np.isnan(numbers).all() else int(np.nanargmin(numbers))

        always_excluded = exclude

        for metric in metrics:
            rule = analysis.metric_info.get(metric, {}).get('best', 'min')
            is_lag = rule == 'zero'
            best_word = {'zero': 'closest to 0', 'max': 'highest'}.get(rule, 'lowest')
            # Persistence's lag is always exactly the horizon (zero spread), so it can't win a lag spread mark
            spread_exclude = {'persistence'} if is_lag else frozenset()
            file_name = metric.lower().replace(' ', '_')
            if 'box' in kinds:
                # Box plot of the per-dataset values (seeds averaged), points on top
                fig = Figure(figsize=(6.5, 8))
                ax = fig.subplots()
                for c in conditions:
                    t, p = c
                    subset = values[(values['t'] == t) & (values['phases'] == p)]
                    model_values = [subset.loc[subset['model_type'] == name, metric].dropna().to_numpy() for name in models]
                    positions = np.arange(len(models)) + offsets[c]
                    boxes = ax.boxplot(model_values, positions=positions, widths=height * 0.85, vert=False,
                                       patch_artist=True, showfliers=False, manage_ticks=False,
                                       medianprops=dict(color='#0b0b0b', lw=1.2))
                    for box in boxes['boxes']:
                        box.set(facecolor=to_rgba(colours[c], 0.35), edgecolor=colours[c], lw=1)
                    for line in boxes['whiskers'] + boxes['caps']:
                        line.set(color=colours[c], lw=0.8)
                    for position, v in zip(positions, model_values):
                        ax.scatter(v, position + jitter.uniform(-height * 0.25, height * 0.25, len(v)), s=6,
                                   color=colours[c], edgecolor='none', zorder=3)
                    # Green: lowest median and smallest spread (interquartile range) within this category
                    medians = [np.median(v) if len(v) else np.nan for v in model_values]
                    iqrs = [np.subtract(*np.percentile(v, [75, 25])) if len(v) else np.nan for v in model_values]
                    if (i := lowest(medians, rule)) is not None:
                        boxes['medians'][i].set(color=self.best_c, lw=2.5, zorder=4)
                    if (i := lowest(iqrs, exclude=spread_exclude)) is not None:
                        boxes['boxes'][i].set(edgecolor=self.best_c, lw=2)
                style(ax, metric, f'{best_word} median / smallest IQR\n(within each category)')
                fig.savefig(out_dir / f'{file_name}_boxplot_grouped.png', dpi=150)

            if 'bars' in kinds:
                # Bars of the bootstrap means with bootstrap standard error bars
                fig = Figure(figsize=(6.5, 8))
                ax = fig.subplots()
                for c in conditions:
                    t, p = c
                    keys = [(name, p, t) for name in models]
                    bar_lengths = np.array([means_table[metric].get(k, np.nan) for k in keys], dtype=float)
                    bar_errors = np.array([std_error_table[metric].get(k, np.nan) for k in keys], dtype=float)
                    positions = np.arange(len(models)) + offsets[c]
                    bars = ax.barh(positions, bar_lengths, height=height * 0.85, color=colours[c])
                    ax.errorbar(bar_lengths, positions, xerr=bar_errors, fmt='none', ecolor='#0b0b0b',
                                elinewidth=0.8, capsize=2)
                    # Green: lowest mean (bar outline) and smallest SE (error bar) within this category
                    if (i := lowest(bar_lengths, rule)) is not None:
                        bars[i].set(edgecolor=self.best_c, lw=2.5)
                    if (i := lowest(bar_errors, exclude=spread_exclude)) is not None:
                        ax.errorbar(bar_lengths[i], positions[i], xerr=bar_errors[i], fmt='none',
                                    ecolor=self.best_c, elinewidth=2.5, capsize=3, zorder=4)
                style(ax, metric, f'{best_word} mean / smallest SE\n(within each category)')
                fig.savefig(out_dir / f'{file_name}_bars_grouped.png', dpi=150)

    @matplotlib.rc_context(PAPER_RC)
    def plot_f1_approaches(self, f1_table: pd.DataFrame,
                           out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\f1') -> None:
        # F1 vs alert window for Torres's four alerting approaches (W, EW, EAW, AW), one figure per (t, phases)
        # category with one panel per approach and one line per model: the mean over the datasets (seeds averaged
        # per dataset) with a +- SE band (1/(B-1), as get_standard_error). Baselines dashed, champions solid.
        # f1_table = metrics_analysis/csv/f1_approaches.csv (analysis.compute_f1_approaches).
        # Saved as <out_dir>/f1_approaches_<t>_<phases>.png
        a = self.analysis
        keys = ['model_type', 'phases', 't', 'approach', 'alert_window']
        per_dataset = f1_table.groupby(keys + ['dataset'])['F1'].mean().reset_index()
        summary = per_dataset.groupby(keys)['F1'].agg(['mean', 'std', 'count']).reset_index()
        summary['se'] = summary['std']          # spread of the per-dataset values, as in the other plots
        models = [m for m in a.baseline_names + a.champion_names if m in set(summary['model_type'])]
        colours = dict(zip(models, self.palette))
        windows = sorted(summary['alert_window'].unique())
        window_labels = {1: '5m', 3: '15m', 6: '30m', 9: '45m', 12: '1h', 24: '2h', 72: '6h'}
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for t in a.times:
            for p in a.phases:
                fig = Figure(figsize=(6.5, 6.5))
                axes = fig.subplots(2, 2, sharex=True, sharey=True)
                for ax, approach in zip(axes.flat, ['W', 'EW', 'EAW', 'AW']):
                    for m in models:
                        s = summary[(summary['model_type'] == m) & (summary['phases'] == p) & (summary['t'] == t)
                                    & (summary['approach'] == approach)].set_index('alert_window').reindex(windows)
                        x = np.arange(len(windows))
                        ls = '--' if m in a.baseline_names else '-'
                        ax.plot(x, s['mean'], ls, color=colours[m], lw=1.5, marker='o', ms=3, label=m)
                        ax.fill_between(x, s['mean'] - s['se'], s['mean'] + s['se'], color=colours[m], alpha=0.12,
                                        lw=0)
                    ax.set_title(approach)
                    ax.set_xticks(np.arange(len(windows)), [window_labels.get(w, str(w)) for w in windows])
                    ax.set_ylim(0, 1)
                    ax.grid(color='#e1e0d9', lw=0.5)
                    for side in ('top', 'right'):
                        ax.spines[side].set_visible(False)
                for ax in axes[1]:
                    ax.set_xlabel('alert window')
                for ax in axes[:, 0]:
                    ax.set_ylabel('F1')
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
                fig.tight_layout(rect=(0, 0, 1, 0.88))
                fig.savefig(out_dir / f'f1_approaches_{t}_{p}.png', dpi=150)

    def plot_train_time_grouped(self, train_times: pd.DataFrame,
                                out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\bootstrap') -> None:
        # Grouped bar chart of the training time (seconds, log axis): mean over the datasets (seeds averaged per
        # dataset, as in plot_train_time) with error bars = SE (1/(B-1)). Persistence isn't trained (~0 s), so it
        # gets no green mark. train_times = metrics_analysis/csv/train_time.csv. Saved as <out_dir>/train_time_bars_grouped.png
        keys = ['model_type', 'phases', 't']
        per_dataset = train_times.groupby(keys + ['dataset'])['Train Time'].mean().reset_index()
        means = per_dataset.groupby(keys)['Train Time'].mean().reset_index()
        std_errors = per_dataset.groupby(keys)['Train Time'].std(ddof=1).reset_index()
        self.plot_mae_mse_grouped((per_dataset,), (means,), (std_errors,), out_dir=out_dir, metrics=('Train Time',),
                                  kinds=('bars',), log_x=True, exclude=frozenset({'persistence'}))

    # ---------- RQ1 summary figures, AGU print style (AGU_RC above) ----------
    # Persistence is the single reference; the other models are grouped empirical (posner) / statistical
    # (linear regression) / machine learning (champions), separated by dashed lines.
    RQ1_METRICS = ('MAE', 'PE', 'ln10 lag', 'F1')
    RQ1_LABELS = {'persistence': 'Persistence', 'posner': 'Posner (2007)', 'linearregression': 'Linear regression',
                  'nn': 'NN', 'rnn': 'RNN', 'rope': 'RoPE Transformer', 'sin': 'Sin Transformer',
                  'zero': 'Zero Transformer', 'tft': 'TFT', 'chronos2': 'Chronos-2'}
    RQ1_METRIC_LABELS = {'MAE': 'MAE', 'PE': 'PE', 'ln10 lag': 'Lag', 'F1': 'F1'}
    RQ1_ARROWS = {'min': '\u2193 better', 'max': '\u2191 better', 'zero': '0 best'}

    def _rq1_setup(self, mean_estimates: tuple[pd.DataFrame], phases: str) -> dict:
        # Per-dataset estimates (seeds averaged) of every model for the RQ1 figures, plus model order and colours.
        # mean_estimates = load_mean_estimates(a) in 'event_avg' mode: (baseline, champion).
        a = self.analysis
        df = pd.concat(mean_estimates)
        df = df[df['phases'] == phases].copy()
        present = set(df['model_type'])
        groups = [[m] for m in ('posner', 'linearregression') if m in present]
        groups.append([m for m in a.champion_names if m in present])
        groups = [g for g in groups if g]
        p = self.palette
        colours = {'persistence': '#0b0b0b', 'posner': p[5], 'linearregression': p[0], 'nn': p[1], 'rnn': p[3],
                   'rope': p[7], 'sin': p[2], 'zero': p[4], 'tft': p[6], 'chronos2': '#777777'}
        best = {m: self.metric_best[m] for m in self.RQ1_METRICS}
        return dict(df=df, groups=groups, models=['persistence'] + [m for g in groups for m in g],
                    colours=colours, best=best, times=list(a.times))

    @staticmethod
    def _rq1_values(df: pd.DataFrame, t: str, model: str, metric: str) -> np.ndarray:
        return df[(df['t'] == t) & (df['model_type'] == model)][metric].dropna().to_numpy()

    def _rq1_cohens_d(self, s: dict, t: str, model: str, metric: str) -> float:
        # (model - persistence) / pooled SD across datasets, signed so positive = better
        def score(v):
            rule = s['best'][metric]
            return -np.abs(v) if rule == 'zero' else (-v if rule == 'min' else v)
        x = score(self._rq1_values(s['df'], t, model, metric))
        r = score(self._rq1_values(s['df'], t, 'persistence', metric))
        if len(x) < 2 or len(r) < 2:
            return np.nan
        pooled = np.sqrt((x.var(ddof=1) + r.var(ddof=1)) / 2)
        return (x.mean() - r.mean()) / pooled if pooled > 0 else 0.0

    def plot_rq1_heatmap(self, mean_estimates: tuple[pd.DataFrame], phases: str = 'nophases',
                         out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\rq1') -> None:
        # One panel per horizon: rows = models, columns = metrics, cell = mean over the bootstrap datasets,
        # colour = Cohen's d vs persistence (blue better, orange worse; colour-blind safe).
        # Saved as <out_dir>/rq1_heatmap_<phases>.pdf (+ .png preview).
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        s = self._rq1_setup(mean_estimates, phases)
        metrics, models, best = self.RQ1_METRICS, s['models'], s['best']
        cmap = LinearSegmentedColormap.from_list('worse_better', [self.palette[5], '#e1e0d9', self.palette[1]])
        cmap.set_bad('#ffffff')
        norm = Normalize(-2, 2)
        breaks = np.cumsum([len(g) for g in s['groups']])[:-1]
        with matplotlib.rc_context(self.AGU_RC):
            fig, axes = plt.subplots(1, len(s["times"]), sharey=True,
                                     figsize=(self.AGU_TWO_COL, (30 + 6 * len(models)) * self.MM))
            axes = np.atleast_1d(axes)
            for k, (ax, t) in enumerate(zip(axes, s['times'])):
                grid = np.full((len(models), len(metrics)), np.nan)
                for i, mod in enumerate(models):
                    for j, m in enumerate(metrics):
                        if mod != 'persistence':
                            grid[i, j] = self._rq1_cohens_d(s, t, mod, m)
                        v = self._rq1_values(s['df'], t, mod, m)
                        ax.text(j, i, f'{v.mean():.2f}' if len(v) else '-', ha='center', va='center', fontsize=7,
                                weight='bold' if mod == 'persistence' else 'normal', color='#0b0b0b')
                im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect='auto')
                ax.set_xticks(range(len(metrics)))
                ax.set_xticklabels([f'{self.RQ1_METRIC_LABELS[m]}\n({self.RQ1_ARROWS[best[m]]})' for m in metrics])
                ax.xaxis.tick_top()
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels([self.RQ1_LABELS.get(m, m) for m in models])
                ax.tick_params(length=0)
                ax.axhline(0.5, color='#0b0b0b', lw=1.0)
                for b in breaks:
                    ax.axhline(b + 0.5, color='#0b0b0b', lw=0.6, ls='--')
                ax.set_xticks(np.arange(-0.5, len(metrics)), minor=True)
                ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
                ax.grid(which='minor', color='white', lw=1.0)
                ax.tick_params(which='minor', length=0)
                for side in ax.spines.values():
                    side.set_visible(False)
                ax.set_xlabel(f'{t} ({int(t[2:]) * 5} min ahead)', labelpad=4)
                self.agu_panel_label(ax, 'abcdefgh'[k], x=-0.01 if k else -0.38, y=1.13)
            cb = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02, extend='both')
            cb.set_label("Cohen's d vs persistence\n(\u2190 worse | better \u2192)")
            cb.outline.set_linewidth(0.5)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f'rq1_heatmap_{phases}.pdf')
            fig.savefig(out / f'rq1_heatmap_{phases}.png', dpi=300)
            plt.close(fig)

    def plot_rq1_box(self, mean_estimates: tuple[pd.DataFrame], phases: str = 'nophases',
                     out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\rq1') -> None:
        # One panel per metric: horizontal box plot (median, quartiles, whiskers 1.5 IQR) per model and horizon,
        # with the per-dataset scores as dots and the persistence medians as vertical lines.
        # Saved as <out_dir>/rq1_box_<phases>.pdf (+ .png preview).
        import matplotlib.pyplot as plt
        s = self._rq1_setup(mean_estimates, phases)
        metrics, best, colours, times = self.RQ1_METRICS, s['best'], s['colours'], s['times']
        ypos, y = {'persistence': -1.4}, 0.0      # reference on top, small gap between model types
        for g in s['groups']:
            for mod in g:
                ypos[mod] = y
                y += 1
            y += 0.4
        offsets = dict(zip(times, np.linspace(-0.19, 0.19, len(times))))
        markers = dict(zip(times, ['o', 's', '^', 'D']))
        line_styles = ['-', ':', '--', '-.']
        rng = np.random.default_rng(0)
        n_cols = 2
        n_rows = int(np.ceil(len(metrics) / n_cols))
        height = min((20 + 8.5 * len(ypos)) * n_rows + 12, 228) * self.MM     # AGU max height 228 mm
        with matplotlib.rc_context(self.AGU_RC):
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(self.AGU_TWO_COL, height), sharey=True)
            axs = axs.ravel()
            for k, (ax, m) in enumerate(zip(axs, metrics)):
                for i, t in enumerate(times):
                    ax.axvline(np.median(self._rq1_values(s['df'], t, 'persistence', m)), color='#0b0b0b', lw=0.6,
                               ls=line_styles[i], zorder=1)
                    for mod in ypos:
                        v = self._rq1_values(s['df'], t, mod, m)
                        if not len(v):
                            continue
                        c, y0 = colours.get(mod, '#777777'), ypos[mod] + offsets[t]
                        ax.boxplot(v, positions=[y0], vert=False, widths=0.3, patch_artist=True, showfliers=False,
                                   boxprops=dict(facecolor=c, alpha=0.35 if i == 0 else 0.15, edgecolor=c, lw=0.6),
                                   medianprops=dict(color=c, lw=1.2), whiskerprops=dict(color=c, lw=0.6),
                                   capprops=dict(color=c, lw=0.6), zorder=2)
                        ax.scatter(v, y0 + rng.uniform(-0.08, 0.08, len(v)), s=5, color=c, edgecolor='#0b0b0b',
                                   lw=0.25, marker=markers[t], zorder=3)
                if best[m] == 'zero':
                    ax.axvline(0, color=self.palette[0], lw=0.5, ls='--')
                for g in s['groups']:
                    ax.axhline(ypos[g[0]] - 0.7, color='#999999', lw=0.5, ls='--')
                ax.grid(axis='x', color='#e1e0d9', lw=0.5)
                ax.set_xlabel(f'{self.RQ1_METRIC_LABELS[m]} ({self.RQ1_ARROWS[best[m]]})')
                ax.tick_params(axis='y', length=0)
                self.agu_panel_label(ax, 'abcdefgh'[k], x=0.0, y=1.02)
            axs[0].set_yticks(list(ypos.values()))
            axs[0].set_yticklabels([self.RQ1_LABELS.get(m, m) for m in ypos])
            axs[0].invert_yaxis()
            for ax in axs[len(metrics):]:
                ax.axis('off')
            handles = [Line2D([], [], marker=markers[t], ls='', color='#888888', mec='#0b0b0b', mew=0.3, ms=4,
                              label=f'{t} ({int(t[2:]) * 5} min), ' + ('darker box' if i == 0 else 'lighter box'))
                       for i, t in enumerate(times)]
            handles += [Line2D([], [], color='#0b0b0b', lw=0.6, ls=line_styles[i], label=f'Persistence median, {t}')
                        for i, t in enumerate(times)]
            fig.tight_layout(w_pad=0.6, h_pad=1.0, rect=(0, 0.04, 1, 1))
            fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.0),
                       handletextpad=0.4, columnspacing=1.2)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f'rq1_box_{phases}.pdf')
            fig.savefig(out / f'rq1_box_{phases}.png', dpi=300)
            plt.close(fig)

    @staticmethod
    def _adjust_p(p: np.ndarray, method: str = 'bonferroni') -> np.ndarray:
        # Multiple-comparison adjusted p-values, capped at 1.
        # 'bonferroni': p x number of tests. 'holm': Holm-Bonferroni step-down (less strict, same error control).
        p = np.asarray(p, dtype=float)
        if method == 'bonferroni':
            return np.minimum(p * len(p), 1.0)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        for k, i in enumerate(order):
            running = max(running, (len(p) - k) * p[i])
            adjusted[i] = min(running, 1.0)
        return adjusted

    def _rq1_tests(self, s: dict, test: str, correction: str = 'bonferroni') -> pd.DataFrame:
        # Every model vs persistence on every RQ1 metric: Cohen's d (positive = better) and two-sided p-value,
        # adjusted ('bonferroni' or 'holm') over all models x metrics within each horizon -> column p_adj.
        # test = 'welch' (independent samples, as in the RQ1 hypothesis tests) or 'paired' (paired t-test on the
        # bootstrap datasets both models were scored on).
        from scipy import stats
        rows = []
        for t in s['times']:
            for mod in s['models'][1:]:
                for m in self.RQ1_METRICS:
                    rule = s['best'][m]
                    sub = s['df'][(s['df']['t'] == t) & s['df']['model_type'].isin([mod, 'persistence'])]
                    wide = sub.pivot_table(index='dataset', columns='model_type', values=m)
                    if mod not in wide or 'persistence' not in wide:
                        continue
                    if rule == 'zero':
                        wide = wide.abs()           # lag: closeness to 0
                    x, r = wide[mod].dropna(), wide['persistence'].dropna()
                    if len(x) < 2 or len(r) < 2:
                        continue
                    if test == 'paired':
                        both = wide[[mod, 'persistence']].dropna()
                        p = stats.ttest_rel(both[mod], both['persistence']).pvalue if len(both) > 1 else np.nan
                    else:
                        p = stats.ttest_ind(x, r, equal_var=False).pvalue
                    rows.append(dict(t=t, model=mod, metric=m, d=self._rq1_cohens_d(s, t, mod, m), p=p))
        tests = pd.DataFrame(rows)
        tests['p_adj'] = np.nan
        for t, idx in tests.groupby('t').groups.items():
            ok = tests.loc[idx, 'p'].notna()
            tests.loc[ok[ok].index, 'p_adj'] = self._adjust_p(tests.loc[ok[ok].index, 'p'].to_numpy(), correction)
        return tests

    def plot_rq1_volcano(self, mean_estimates: tuple[pd.DataFrame], phases: str = 'nophases', test: str = 'welch',
                         correction: str = 'bonferroni',
                         out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\rq1') -> None:
        # Volcano plot, one panel per horizon: each point is one model x metric, x = Cohen's d vs persistence
        # (positive = better), y = -log10(adjusted two-sided p; Bonferroni by default, or Holm).
        # Colour = model, marker = metric.
        # Dashed lines at p = 0.05 and 0.01. Upper right = significantly and substantially better than persistence.
        # Saved as <out_dir>/rq1_volcano_<phases>.pdf (+ .png preview) and the numbers as rq1_tests_<phases>.csv.
        import matplotlib.pyplot as plt
        s = self._rq1_setup(mean_estimates, phases)
        tests = self._rq1_tests(s, test, correction)
        markers = dict(zip(self.RQ1_METRICS, ['o', 's', '^', 'D', 'v']))
        tests['logp'] = -np.log10(tests['p_adj'].clip(lower=1e-12))
        y_top = max(3.0, np.nanmax(tests['logp']) * 1.08)
        x_lim = max(2.5, np.nanmax(np.abs(tests['d'])) * 1.08)
        with matplotlib.rc_context(self.AGU_RC):
            fig, axes = plt.subplots(1, len(s['times']), sharey=True, figsize=(self.AGU_TWO_COL, 72 * self.MM))
            axes = np.atleast_1d(axes)
            for k, (ax, t) in enumerate(zip(axes, s['times'])):
                ax.axhspan(-np.log10(0.05), y_top, xmin=0.5, xmax=1, color=self.palette[1], alpha=0.06, lw=0)
                ax.axhspan(-np.log10(0.05), y_top, xmin=0, xmax=0.5, color=self.palette[5], alpha=0.06, lw=0)
                # Significance threshold p = 0.05, with the two regions labelled
                p_y = -np.log10(0.05)
                ax.axhline(p_y, color='#0b0b0b', lw=0.6, ls='--')
                ax.text(x_lim * 0.97, p_y + 0.08, 'significant (p < 0.05)', fontsize=6, va='bottom', ha='right')
                ax.text(x_lim * 0.97, p_y - 0.08, 'not significant', fontsize=6, va='top', ha='right')
                ax.axvline(0, color='#0b0b0b', lw=0.6)
                # Large-effect cut-off |d| = 0.8 (Cohen), the same for every metric since d is standardised
                for sign in (-1, 1):
                    ax.axvline(sign * 0.8, color='#888888', lw=0.6, ls='-.')
                sub = tests[tests['t'] == t]
                for _, r in sub.iterrows():
                    ax.scatter(r['d'], r['logp'], marker=markers[r['metric']], s=22,
                               color=s['colours'].get(r['model'], '#777777'), edgecolor='#0b0b0b', lw=0.4, zorder=3)
                ax.set_xlim(-x_lim, x_lim)
                ax.set_ylim(0, y_top)
                ax.grid(color='#e1e0d9', lw=0.5)
                ax.set_xlabel("Cohen's d vs persistence  (\u2190 worse | better \u2192)")
                ax.set_title(f'{t} ({int(t[2:]) * 5} min ahead)')
                self.agu_panel_label(ax, 'abcdefgh'[k], x=0.0, y=1.02)
            axes[0].set_ylabel(r'$-\log_{10}$(' + correction.capitalize() + '-adjusted p)')
            model_handles = [Line2D([], [], marker='o', ls='', color=s['colours'].get(m, '#777777'), mec='#0b0b0b',
                                    mew=0.4, ms=5, label=self.RQ1_LABELS.get(m, m)) for m in s['models'][1:]]
            metric_handles = [Line2D([], [], marker=markers[m], ls='', color='#bbbbbb', mec='#0b0b0b', mew=0.4, ms=5,
                                     label=self.RQ1_METRIC_LABELS[m]) for m in self.RQ1_METRICS]
            metric_handles += [Line2D([], [], color='#888888', lw=0.6, ls='-.', label='|d| = 0.8 (large effect)')]
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            fig.legend(handles=model_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.055),
                       handletextpad=0.3, columnspacing=1.0)
            fig.legend(handles=metric_handles, loc='lower center', ncol=len(metric_handles),
                       bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f'rq1_volcano_{phases}.pdf')
            fig.savefig(out / f'rq1_volcano_{phases}.png', dpi=300)
            plt.close(fig)
            tests.drop(columns='logp').to_csv(out / f'rq1_tests_{phases}.csv', index=False, float_format='%.4g')

    # ---------- RQ2 figure, AGU print style ----------
    def plot_rq2_warning(self, warning_times: pd.DataFrame, phases: str = 'nophases', show_bands: bool = False,
                         out_dir: str = r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\rq2') -> None:
        # Share of SEP events warned at least R minutes before the 10 pfu crossing, against the assumed satellite
        # response time R; one panel per horizon, one line per model. Per bootstrap dataset the share is averaged over
        # seeds; line = mean over datasets (show_bands: + 2.5-97.5 % range over datasets). Shaded: electron head start
        # (~17-55 min); dotted vertical: the forecast horizon.
        # warning_times = analysis.compute_warning_times(...). Saved as <out_dir>/rq2_warning_<phases>.pdf (+ .png)
        import matplotlib.pyplot as plt
        a = self.analysis
        df = warning_times[warning_times['phases'] == phases]
        present = set(df['model_type'])
        order = ['persistence'] + [m for m in ('posner', 'linearregression') if m in present] + \
                [m for m in a.champion_names if m in present]
        order = [m for m in order if m in present]
        p = self.palette
        colours = {'persistence': '#0b0b0b', 'posner': p[5], 'linearregression': p[0], 'nn': p[1], 'rnn': p[3],
                   'rope': p[7], 'sin': p[2], 'zero': p[4], 'tft': p[6], 'chronos2': '#777777'}
        R = np.arange(5, 125, 5)
        with matplotlib.rc_context(self.AGU_RC):
            fig, axes = plt.subplots(1, len(a.times), sharey=True, figsize=(self.AGU_TWO_COL, 70 * self.MM))
            axes = np.atleast_1d(axes)
            for k, (ax, t) in enumerate(zip(axes, a.times)):
                h_min = int(t[2:]) * 5
                ax.axvspan(17, 55, color='#e1e0d9', alpha=0.6, lw=0, zorder=0)
                ax.axvline(h_min, color='#0b0b0b', lw=0.6, ls=':', zorder=1)
                for mod in order:
                    m = df[(df['t'] == t) & (df['model_type'] == mod)]
                    if m.empty:
                        continue
                    curves = np.array([[(d['warning_min'] >= r).mean() for r in R]
                                       for _, d in m.groupby('dataset')])
                    c = colours.get(mod, '#777777')
                    ls = '--' if mod in a.baseline_names else '-'
                    ax.plot(R, curves.mean(axis=0), ls, color=c, lw=1.2, label=self.RQ1_LABELS.get(mod, mod),
                            zorder=3)
                    if show_bands and len(curves) > 1:
                        lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
                        ax.fill_between(R, lo, hi, color=c, alpha=0.10, lw=0, zorder=2)
                ax.set_xlim(R[0], R[-1])
                ax.set_ylim(0, 1)
                ax.set_xlabel('Assumed satellite response time R (min)')
                ax.set_title(f'{t} ({h_min} min ahead)')
                ax.grid(color='#e1e0d9', lw=0.5)
                self.agu_panel_label(ax, 'abcdefgh'[k], x=0.0, y=1.02)
            axes[0].set_ylabel('Share of events warned \u2265 R ahead')
            handles, labels = axes[0].get_legend_handles_labels()
            handles += [plt.Rectangle((0, 0), 1, 1, color='#e1e0d9', alpha=0.6),
                        Line2D([], [], color='#0b0b0b', lw=0.6, ls=':')]
            labels += ['Electron head start (~17\u201355 min)', 'Forecast horizon']
            fig.tight_layout(rect=(0, 0.16, 1, 1))
            fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0),
                       handletextpad=0.4, columnspacing=1.0)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f'rq2_warning_{phases}.pdf')
            fig.savefig(out / f'rq2_warning_{phases}.png', dpi=300)
            plt.close(fig)
