import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotStuff:
    # Takes an Analysis object and reads its settings (metrics, times, baseline names,
    # results folder) from it, so the two classes never disagree.
    def __init__(self, analysis, show: bool = False):
        self.analysis = analysis
        self.show     = show   # open the histogram booklet window (blocks until closed); PNGs are saved either way
        # "Bright" palette from simplifiedsciencepublishing.com, in the order listed there.
        self.palette  = ['#003a7d', '#008dff', '#ff73b6', '#c701ff',
                         '#4ecb8d', '#ff9d3a', '#f9e858', '#d83034']
        self.best_c   = self.palette[4]   # green border on the info box of the best model
        # Each event gets its own colour + marker (8 colours x 5 shapes = 40 styles for the 39 events),
        # fixed by event number so an event looks the same in every plot.
        self.event_markers = ['o', 's', '^', 'D', 'v']

        # What "best" means for each metric and its per-event folder, from the one table in Analysis.metric_info
        self.metric_best = {m: info['best'] for m, info in analysis.metric_info.items()}

        # Metric histograms: one folder per metric, e.g. metrics_analysis/hists/phases/mae/
        self.metric_folders = {m: info['folder'] for m, info in analysis.metric_info.items()}
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

    def plot_hypothesis_grid(self, mean_test: pd.DataFrame) -> None:
        # One PNG per champion: a reject / don't-reject grid against each baseline (see Analysis.perform_mean_test).
        # Rows = metrics, columns = t+6 / t+12 x phases / no phases. Each cell shows the one-sided p-value.
        # Green = reject H0 (the champion is significantly better), red = don't reject, grey = not enough data.
        # Saved as results/metrics_analysis/hypothesis_grids/<champion>.png
        from matplotlib.patches import Rectangle, FancyBboxPatch
        a = self.analysis
        colours = {'reject': '#0ca30c', 'keep': '#d03b3b', 'na': '#e1e0d9'}
        metrics = [m for m in a.results_metrics if m in set(mean_test['metric'])]
        cols = [(t, p) for t in a.times for p in a.phases]
        col_labels = [f"{t}\n{'phases' if p == 'phases' else 'no phases'}" for t, p in cols]
        baselines = [b for b in a.baseline_names if b in set(mean_test['baseline'])]
        h_a = mean_test.drop_duplicates('metric').set_index('metric')['h_a']
        cells = mean_test.set_index(['model_type', 'baseline', 'metric', 't', 'phases'])
        alpha = a.significance_level

        out_dir = a.results_folder / 'metrics_analysis' / 'hypothesis_grids'
        out_dir.mkdir(parents=True, exist_ok=True)
        for champ in [c for c in a.champion_names if c in set(mean_test['model_type'])]:
            fig = Figure(figsize=(12, 3.0 * len(baselines) + 1.0), dpi=100)
            fig.suptitle(f'{champ} vs baselines: one-sided Welch t-test on the per-dataset means, alpha = {alpha}',
                         fontsize=13)
            gs = fig.add_gridspec(len(baselines), 2, width_ratios=[1, 2.4], wspace=0.25, hspace=0.45)

            # Left: the hypotheses, as in the sketch
            info = fig.add_subplot(gs[:, 0]); info.axis('off')
            short = lambda h: h.replace('champion', 'c').replace('baseline', 'b')
            width = max(len(m) for m in metrics)
            lines = ['H0: champion no better', 'HA: (reject => champion better)', '']
            lines += [f'{m:<{width}}  {short(h_a[m])}' for m in metrics]
            lines += ['', 'c = champion, b = baseline', 'lags: distance from 0',
                      'p = one-sided p-value']
            info.text(0.02, 0.5, '\n'.join(lines), va='center', ha='left', fontsize=9, family='monospace',
                      bbox=dict(boxstyle='round,pad=0.8', fc='#f7f7f5', ec='#c3c2b7'))
            handles = [Rectangle((0, 0), 1, 1, fc=colours[k]) for k in ('reject', 'keep', 'na')]
            info.legend(handles, ['Reject H0 (champion better)', "Don't reject H0", 'Not enough data'],
                        loc='lower left', fontsize=9, frameon=False)

            # Right: one grid per baseline
            for i, bl in enumerate(baselines):
                ax = fig.add_subplot(gs[i, 1])
                ax.set_title(f'vs {bl}', fontsize=11)
                for r, m in enumerate(metrics):
                    for c, (t, p) in enumerate(cols):
                        key = (champ, bl, m, t, p)
                        pv = cells['p_value'].get(key, np.nan) if key in cells.index else np.nan
                        state = 'na' if pd.isna(pv) else 'reject' if pv < alpha else 'keep'
                        ax.add_patch(FancyBboxPatch((c + 0.04, r + 0.06), 0.92, 0.88,
                                                    boxstyle='round,pad=0,rounding_size=0.06',
                                                    fc=colours[state], ec='none'))
                        word = {'reject': 'Reject', 'keep': "Don't reject", 'na': 'n/a'}[state]
                        text = word if state == 'na' else f'{word}\np = {pv:.3g}'
                        ax.text(c + 0.5, r + 0.5, text, ha='center', va='center', fontsize=8,
                                color='#0b0b0b' if state == 'na' else 'white')
                ax.set_xlim(0, len(cols)); ax.set_ylim(len(metrics), 0)
                ax.set_xticks(np.arange(len(cols)) + 0.5, col_labels, fontsize=9)
                ax.set_yticks(np.arange(len(metrics)) + 0.5, metrics, fontsize=9)
                ax.xaxis.tick_top()
                ax.tick_params(length=0)
                for side in ax.spines.values():
                    side.set_visible(False)

            fig.savefig(out_dir / f'{champ}.png', dpi=120, bbox_inches='tight')
