from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import re
from scipy.stats import t
from core.plotstuff import PlotStuff
from lib.utils.events import read_event_timestamps, n_test_samples, find_test_events
from lib.utils.paths import parse_results_path

class Analysis:
    def __init__(self, results_folder: str = "results", train_split: float = .8, significance_level: float = .05):
        self.results_folder        = Path(results_folder)
        self.baseline_names        = ['linearregression', 'posner', 'persistence']
        self.champion_names        = ['nn', 'rnn', 'rope', 'sin', 'zero', 'tft', 'chronos2']
        self.baseline_h0           = 'persistence' # TODO read from config?
        self.model_names_w_results = self.get_model_names_w_results()
        self.desired_results_metrics       = ['MAE', 'O2P lag', 'O2T lag', 'ln10 lag', 'PE', 'F1']
        self.phases                = ['phases', 'nophases']
        self.file_type             = ['results', 'predictions']
        self.times                 = ['t+6', 't+12']
        self.train_split           = train_split
        self.true_protons          = self.get_true_protons()
        self.significance_level    = significance_level
        self.results_metrics       = self.desired_results_metrics                    # "Average ..." lines of results.txt
        self.event_metrics         = ['MAE', 'O2P lag', 'O2T lag', 'ln10 lag', 'PE', 'F1']  # "Event i" blocks (r2 = PE per event)
        # The one metric table: what "best" means (lowest, highest, closest to 0) and each metric's plot folder.
        self.metric_info = {
            'MAE':        {'best': 'min',  'folder': 'mae'},
            'O2P lag':    {'best': 'zero', 'folder': 'o2p'},
            'O2T lag':    {'best': 'zero', 'folder': 'o2t'},
            'ln10 lag':   {'best': 'zero', 'folder': 'ln10lag'},
            'PE':         {'best': 'max',  'folder': 'pe'},
            'r2':         {'best': 'max',  'folder': 'r2'},
            'F1':         {'best': 'max',  'folder': 'f1'},
            'Train Time': {'best': 'min',  'folder': 'train_time'},
        }
        self.set_bootstrap_mode('event_avg')

    def set_bootstrap_mode(self, mode: str) -> None:
        # Which values the bootstrap functions below work on, grouped by what, and where they save:
        #   'event_avg': one value per run = the "Average ..." lines of results.txt (averaged over a dataset's events).
        #                Bootstrapped over datasets per (model, phases, t) -> results/metrics_analysis/csv/<baseline|champion>/event_avg/
        #   'events':    one value per run per event = the "Event i" blocks of results.txt.
        #                Bootstrapped over the datasets (and seeds) that contain the event, per (model, phases, t, event)
        #                -> results/metrics_analysis/csv/<baseline|champion>/events/
        # bootstrap_keys: what one bootstrap estimate describes. estimate_keys: what one bootstrap sample is.
        self.bootstrap_mode = mode
        self.metrics_folder = 'metrics_analysis/csv'     # bootstrap CSVs
        self.hists_folder   = 'metrics_analysis/hists'   # histogram PNGs: hists/<phases>/<metric>/
        if mode == 'event_avg':
            self.desired_results_metrics = self.results_metrics
            self.bootstrap_keys = ['model_type', 'phases', 't']
            self.estimate_keys  = ['model_type', 'phases', 'dataset', 't']                    # seeds averaged per dataset
        elif mode == 'events':
            self.desired_results_metrics = self.event_metrics
            self.bootstrap_keys = ['model_type', 'phases', 't', 'event']
            self.estimate_keys  = ['model_type', 'phases', 'dataset', 't', 'event', 'run']    # seeds pooled
        else:
            raise ValueError(f"unknown bootstrap mode {mode!r} (use 'event_avg' or 'events')")

    def csv_path(self, group: str, name: str) -> Path:
        # Bootstrap CSVs: results/metrics_analysis/csv/<baseline|champion>/<event_avg|events>/<name>.csv
        return self.results_folder / self.metrics_folder / group / self.bootstrap_mode / f'{name}.csv'

    def save_csv(self, df: pd.DataFrame, name: str, index: bool = False, folder: str | None = None,
                 float_format: str = '%.4f') -> None:
        # Default: csv_path(...) for the current bootstrap mode. With folder given (e.g. 'true_vs_pred'):
        # results/<folder>/<baseline|champion>_<name>.csv
        group = 'baseline' if df['model_type'].iloc[0] in self.baseline_names else 'champion'
        path = self.csv_path(group, name) if folder is None else self.results_folder / folder / f'{group}_{name}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index, float_format=float_format)

    def save_test_csv(self, df: pd.DataFrame, name: str) -> None:
        # Hypothesis test tables (all champion x baseline pairs in one file):
        # results/metrics_analysis/<name>_<event_avg|events>.csv. %.6g so small p-values are not rounded to 0.
        path = self.results_folder / 'metrics_analysis' / f'{name}_{self.bootstrap_mode}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format='%.6g')

    def get_model_names_w_results(self)-> list[str]:
        known = set(self.baseline_names + self.champion_names)
        return [mrf.name for mrf in self.results_folder.iterdir() if mrf.is_dir() and mrf.name in known]

    def read_predictions_file(self, paths: pd.Series)-> np.array:
        all_data = []
        for path in paths:
            predictions = np.loadtxt(path)
            all_data.append(predictions)
        prediction_block=np.array(all_data).T   # (time steps, runs); reshape would interleave the runs


        return prediction_block

    def get_files_by_model(self,file_type: str)->pd.DataFrame:
        baseline_data = []
        champion_data = []

        for mn in self.model_names_w_results:
            for p in self.phases:
                for path in (self.results_folder / mn / p).rglob("*.txt"):
                    # root, model_type, phases, dataset, t, [model_name,] file, path
                    path_parts = parse_results_path(self.results_folder, path)

                    if mn in self.baseline_names:
                        baseline_data.append(path_parts)
                    else:
                        champion_data.append(path_parts)

        # print(baseline_data)
        b_df = pd.DataFrame(baseline_data, columns = ['root', 'model_type', 'phases','dataset','t','file', 'path'])
        c_df = pd.DataFrame(champion_data, columns = ['root', 'model_type', 'phases','dataset','t','model_name','file', 'path'])

        b_df = b_df[b_df["file"] == file_type + ".txt"]
        c_df = c_df [c_df["file"] == file_type + ".txt"]

        return b_df, c_df
    
    def pull_metrics_from_results_file(self, path: Path, desired_metric:str)-> dict[str, any]:

        with open(path, "r") as f:
            for line in reversed(f.readlines()):      
                metric_line = line.strip().split("=")[0].strip() 

                if desired_metric in metric_line:
                    line =line.strip().split("=")
                    if 'Train Time' in metric_line:

                        return pd.to_timedelta(line[1]).total_seconds()

                    else:
                        return float(line[1])


    def compile_bootstrap_data(self, file_type: str) -> tuple[pd.DataFrame]:
        files = self.get_files_by_model(file_type)
        if file_type == 'results':
                
            for f in files:
                for dm in self.desired_results_metrics:
                    
                    f[dm] = f['path'].apply(lambda p: self.pull_metrics_from_results_file(p,dm)) 


                self.save_csv(f, f'bootstrap_{file_type}', index=True)
            return files
        else:
            for f in files:
                # list of prediction files: comes from the predictions, so it lives with the true vs pred output
                self.save_csv(f, f'bootstrap_{file_type}', index=True, folder='true_vs_pred')
            return files



    def pull_events_from_results_file(self, path: Path) -> dict[int, dict[str, float]]:
        # {i: {metric: value}} for every "Event i" block in a results.txt (i is 1-based, in file order)
        text = open(path).read()
        blocks = re.findall(r'^Event (\d+)\n((?:.+\n)+?)\n', text, re.M)
        return {int(i): {k: float(v) for k, v in re.findall(r'^(.+?) =\s*(\S+)$', body, re.M)} for i, body in blocks}

    def compile_event_bootstrap_data(self) -> tuple[pd.DataFrame]:
        # Like compile_bootstrap_data('results'), but one row per event: the "Event i" blocks instead of the
        # "Average ..." lines. "Event i" is the i-th test event of that dataset in event_timestamps.txt order,
        # i.e. the i-th key of self.event_slices[t][ds] (same rule as evaluate()).
        tables = []
        for files in self.get_files_by_model('results'):
            rows = []
            for _, f in files.iterrows():
                events = sorted(self.event_slices[f['t']][f['dataset']], key=lambda e: int(e[1:]))
                blocks = self.pull_events_from_results_file(f['path'])
                if len(blocks) != len(events):
                    print(f"WARNING: {f['path']} has {len(blocks)} events, expected {len(events)}; skipped")
                    continue
                run = f['model_name'] if 'model_name' in f else 'run'
                for i, e in enumerate(events, start=1):
                    rows.append({'model_type': f['model_type'], 'phases': f['phases'], 'dataset': f['dataset'],
                                 't': f['t'], 'run': run, 'event': e,
                                 **{m: blocks[i].get(m, np.nan) for m in self.event_metrics}})
            df = pd.DataFrame(rows)
            self.save_csv(df, 'bootstrap_results', index=True)   # -> csv/<group>/events/bootstrap_results.csv
            tables.append(df)
        return tuple(tables)

    def get_train_times(self) -> pd.DataFrame:
        # Train Time of every run (seconds), one row per model / phases / dataset / t / seed.
        # It is one number per run, not per event, so it gets its own table: metrics_analysis/csv/train_time.csv
        rows = []
        for files in self.get_files_by_model('results'):
            for _, f in files.iterrows():
                rows.append({'model_type': f['model_type'], 'phases': f['phases'], 'dataset': f['dataset'], 't': f['t'],
                             'run': f['model_name'] if 'model_name' in f else 'run',
                             'Train Time': self.pull_metrics_from_results_file(f['path'], 'Train Time')})
        train_times = pd.DataFrame(rows)
        out_dir = self.results_folder / self.metrics_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        train_times.to_csv(out_dir / 'train_time.csv', index=False, float_format='%.4f')
        return train_times

    def get_predictions_by_dataset(self, data:tuple[pd.DataFrame])-> tuple[dict]:

        all_data = []
        prediction_datakeys = []
        for d in data:
            predictions = {}

            model_predictions_groups  =  d.groupby(["model_type", "t","phases",'dataset'])
            current_datakeys = list(model_predictions_groups.groups.keys())
            prediction_datakeys += current_datakeys

            block_of_dataset_predictions = model_predictions_groups['path'].apply(lambda x: self.read_predictions_file(x))

            predictions['data'] = {}

            for  i, (mt, t, p, ds) in enumerate(current_datakeys):
                    # one entry per event in this dataset's test split: rows onset -> peak, shape (n_e, runs)
                    raw = block_of_dataset_predictions.values[i]
                    for e, event_rows in self.event_slices[t][ds].items():
                        predictions['data'].setdefault(mt, {}).setdefault(t, {}).setdefault(p, {}).setdefault(ds, {})[e] = raw[event_rows]

            all_data.append(predictions)
        # keys are now (model_type, t, phases, dataset, event), e.g. ('nn', 't+6', 'phases', 'dataset_3', 'e22')
        self.prediction_datakeys = {(mt, t, p, ds, e) for (mt, t, p, ds) in prediction_datakeys
                                    for e in self.event_slices[t][ds]}
        return tuple(all_data)
    
    # TODO: create function to extend prediction database. 
    #  - takes in pred/true data
    #  - function to act on data (mean, ssx, b1, b0, sse etc), 
    #   - name of new extension
    #  -Do it by events in dataset not by dataset
    # 
    def extend_prediction_database(self, pred_data: tuple[dict], extension_name:str, extension_action: callable, *sources:str) -> tuple[dict]:
        all_data = []
        for d in pred_data:
            d[extension_name] = {}
            for (mt, t, p, ds, e) in self.prediction_datakeys:
                try:
                    # 'true' is this event's true protons from self.true_protons, as an (n_e, 1) column
                    inputs = [self.true_protons[t][ds][e][:, None] if s == 'true' else d[s][mt][t][p][ds][e]
                              for s in sources]
                    extension_value = extension_action(*inputs)

                    d[extension_name].setdefault(mt, {}).setdefault(t, {}).setdefault(p, {}).setdefault(ds, {})[e] = extension_value
                except KeyError:
                    pass
            all_data.append(d)
        return tuple(all_data)

    def get_event_timestamps(self) -> list[list[str]]:
        # Each line: onset, threshold, peak, end (same reader as SetUp)
        return read_event_timestamps(str(Path("data") / "event_timestamps.txt"))

    def get_true_protons(self)-> dict[str, dict[str, dict[str, np.ndarray]]]:
        # true_protons[t][ds][e] = the true protons of event e in dataset ds, onset -> peak, shape (n_e,).
        # Also sets self.event_slices[t][ds][e], that event's rows within the dataset's test predictions.
        # Keys: ds = 'dataset_3', e = 'e22' (line 22 of event_timestamps.txt).
        #
        # Test rows mirror BaselineModel.pair_input_output: samples t in range(24, len - h), split
        # 80/20 on samples, so the test targets are the last n_test rows.
        # Events are found by timestamp, not event_indices.txt: those indices are for data_0 only,
        # and each bootstrap dataset shuffles the events to different rows. Same rule as evaluate():
        # an event counts if its onset is in the test targets, and is scored from onset to peak.
        # Events that are never in a test split (training data in every dataset) don't appear.
        files = sorted(Path("data").glob("data_*.csv"), key=lambda f: int(f.stem.split('_')[1]))
        frames = {f"dataset_{f.stem.split('_')[1]}": pd.read_csv(f, usecols=["time", "proton"]) for f in files}
        events = self.get_event_timestamps()
        self.event_onsets = {f"e{i + 1}": ev[0] for i, ev in enumerate(events)}

        true_protons, self.event_slices = {}, {}
        for t in self.times:
            h = int(t[2:])
            true_protons[t], self.event_slices[t] = {}, {}
            for ds, df in frames.items():
                # same split and same event rule as the models (lib/utils/events.py)
                n_test = n_test_samples(len(df), h, self.train_split)
                test_times = df["time"].to_numpy()[-n_test:]
                test_protons = df["proton"].to_numpy()[-n_test:]

                true_protons[t][ds], self.event_slices[t][ds] = {}, {}
                for i, (onset_row, _threshold_row, peak_row) in find_test_events(test_times, events).items():
                    e = f"e{i + 1}"
                    rows = slice(onset_row, peak_row + 1)
                    self.event_slices[t][ds][e] = rows
                    true_protons[t][ds][e] = test_protons[rows]
        return true_protons
    
    def get_prediction_mean(self, raw:np.array)->np.array:
        return raw.mean(axis=0)

    def get_prediction_ssx(self, raw:np.array, mean:np.array)->np.array:
        return ((raw - mean) ** 2).sum(axis=0)

    def get_prediction_b1(self, raw:np.array, mean:np.array, ssx:np.array, true:np.array)->np.array:
        return ((raw - mean) * true).sum(axis=0) / ssx # name b1

    def get_prediction_b0(self, mean:np.array, b1:np.array, true:np.array)->np.array:
        return true.mean(axis = 0) - b1 * mean #name b0

    def get_prediction_sse(self, raw:np.array, b0: np.array, b1:np.array, true:np.array)-> np.array:
        return (true**2).sum(axis = 0) - b0 * true.sum(axis = 0) - b1 * (raw * true).sum(axis = 0) #name sse

    def get_prediction_ssr(self, raw:np.array, true:np.array)-> np.array:
        # residual sum of squares of the raw predictions (not of the fitted line)
        return ((true - raw)**2).sum(axis = 0) #name ssr

    def get_prediction_sst(self, true:np.array)->np.array:
        return ((true - true.mean(axis = 0))**2).sum(axis = 0) #name sst

    def get_prediction_r2(self, ssr:np.array, sst:np.array)-> np.array:
        return 1 - (ssr/sst) #name r2

    def get_prediction_sample_variance(self, raw:np.array, sse:np.array)-> np.array:
        return sse/ (raw.shape[0] - 2) #name s2

    def get_prediction_b1_std_err(self, s2:np.array, sxx: np.array)-> np.array:
        return np.sqrt(s2/sxx) #name b1_std_err

    def get_prediction_b0_std_err(self, raw:np.array, mean:np.array, ssx:np.array, s2:np.array)-> np.array:
        return np.sqrt(s2 * (1 / raw.shape[0] + mean**2 / ssx)) #name b0_std_err

    def get_prediction_t_critical_values(self, raw:np.array)-> np.array:
        return t.ppf((1 - self.significance_level / 2), raw.shape[0] - 2) #name t_crit

    def get_prediction_confidence_intervals(self, b1:np.array, b1_std_err:np.array, t_crit:np.array) ->np.array:
        l_bound = b1 - t_crit * b1_std_err
        u_bound = b1 + t_crit * b1_std_err
        return (l_bound, u_bound) # name ci
        
    def get_prediction_predictions_intervals(self, raw:np.array, mean:np.array, ssx:np.array, s2:np.array, t_crit:np.array, b0:np.array, b1:np.array) ->np.array:
        n = raw.shape[0]

        true_est = b0+b1*raw
        l_bound = true_est - t_crit * np.sqrt(s2 * (1 + 1/n + (raw- mean)**2 / ssx))
        u_bound = true_est + t_crit * np.sqrt(s2 * (1 + 1/n + (raw- mean)**2 / ssx))
        return (l_bound, u_bound) # name pi

    def save_true_vs_pred_stats(self, pred_data: tuple[dict]) -> tuple[pd.DataFrame]:
        # One row per (model_type, phases, t, dataset, run) with every regression value, saved to
        # results/true_vs_pred/{baseline|champion}_true_vs_pred_stats.csv, plus the bootstrap means
        # per (model_type, phases, t) in {baseline|champion}_true_vs_pred_mean_stats.csv.
        # PI: its lines run parallel to the fit, so it's stored as the two intercepts b0 -/+ t*s
        # (pi_b0_lower/upper); pi_within is the share of points inside the exact PI.
        scalars = ['mean', 'ssx', 'b1', 'b1_std_err', 'b0', 'b0_std_err', 'sse', 's2', 'ssr', 'sst', 'r2', 'r2_event_mean']
        tables = []
        for d in pred_data:
            rows, event_rows = [], []
            for (mt, t, p, ds) in sorted(self.prediction_datakeys):
                try:
                    raw = d['data'][mt][t][p][ds]
                    v = {name: d[name][mt][t][p][ds] for name in scalars}
                    t_crit = d['t_crit'][mt][t][p][ds]
                    ci_b1, ci_b0 = d['ci'][mt][t][p][ds], d['ci_b0'][mt][t][p][ds]
                except KeyError:
                    continue

                true = self.true_protons[t][int(ds.split('_')[1])][:, None]
                pi_lo, pi_hi = self.get_prediction_predictions_intervals(
                    raw, v['mean'], v['ssx'], v['s2'], t_crit, v['b0'], v['b1'])
                pi_within = ((true >= pi_lo) & (true <= pi_hi)).mean(axis=0)
                pi_within = np.where(np.isnan(v['s2']), np.nan, pi_within)   # NaN fit -> NaN, not 0% inside
                pi_half = t_crit * np.sqrt(v['s2'])

                # one row per (run, event) for the per-event csv
                k = int(ds.split('_')[1])
                event_r2 = d['event_r2'][mt][t][p][ds]
                for i, e in enumerate(np.unique(self.event_ids[t][k])):
                    n_e = int((self.event_ids[t][k] == e).sum())
                    for run in range(raw.shape[1]):
                        event_rows.append({'model_type': mt, 'phases': p, 't': t, 'dataset': ds, 'run': run,
                                           'event': int(e), 'onset': self.event_onsets[e], 'n': n_e,
                                           'r2': event_r2[i][run]})

                for run in range(raw.shape[1]):
                    row = {'model_type': mt, 'phases': p, 't': t, 'dataset': ds, 'run': run, 'n': raw.shape[0]}
                    # broadcast so one-per-key values (sst has shape (1,)) work like one-per-run values
                    row.update({name: np.broadcast_to(v[name], (raw.shape[1],))[run] for name in scalars})
                    row.update({
                        't_crit': t_crit,
                        'ci_b1_lower': ci_b1[0][run], 'ci_b1_upper': ci_b1[1][run],
                        'ci_b0_lower': ci_b0[0][run], 'ci_b0_upper': ci_b0[1][run],
                        'pi_b0_lower': v['b0'][run] - pi_half[run], 'pi_b0_upper': v['b0'][run] + pi_half[run],
                        'pi_within': pi_within[run],
                    })
                    rows.append(row)
            if not rows:
                continue

            df = pd.DataFrame(rows)
            # more precision than the default %.4f: the standard errors are ~1e-4
            self.save_csv(df, 'true_vs_pred_stats', folder='true_vs_pred', float_format='%.8g')
            mean_df = df.drop(columns=['dataset', 'run']).groupby(['model_type', 'phases', 't'], as_index=False).mean()
            self.save_csv(mean_df, 'true_vs_pred_mean_stats', folder='true_vs_pred', float_format='%.8g')
            # per-event R^2 for every (model, phases, t, dataset, run, event)
            self.save_csv(pd.DataFrame(event_rows), 'true_vs_pred_event_stats', folder='true_vs_pred', float_format='%.8g')
            tables.append(df)
        return tuple(tables)

    def get_mean_estimates(self, data:tuple[pd.DataFrame])-> tuple[pd.DataFrame]:
        mean_estimates = []

        for d in data:
            bootstrap_means = pd.DataFrame([])
            bootstrap_group = d.groupby(self.estimate_keys)
            for dm in self.desired_results_metrics:
                        
                bootstrap_mean = bootstrap_group[dm].mean().reset_index()
                missing_columns = bootstrap_mean.columns.difference(bootstrap_means.columns)

                for mc in missing_columns:
                    bootstrap_means = pd.concat([bootstrap_means, bootstrap_mean[mc]], axis=1)



            self.save_csv(bootstrap_means, 'mean_estimates')

            mean_estimates.append(bootstrap_means)
        # print(tuple(means))
        return tuple(mean_estimates)


    def get_mean_bootstrap_estimates(self,estimate_data:tuple[pd.DataFrame]):
        bootstrap_mean_estimates = []
        for ed in estimate_data:
            bootstrap_means = pd.DataFrame([])
            bootstrap_group = ed.groupby(self.bootstrap_keys)

            for dm in self.desired_results_metrics:
                        
                bootstrap_mean = bootstrap_group[dm].mean().reset_index()
                # print(bootstrap_mean)
                missing_columns = bootstrap_mean.columns.difference(bootstrap_means.columns)

                for mc in missing_columns:
                    bootstrap_means = pd.concat([bootstrap_means, bootstrap_mean[mc]], axis=1)



            self.save_csv(bootstrap_means, 'mean_bootstrap_estimates')

            bootstrap_mean_estimates.append(bootstrap_means)
        return tuple(bootstrap_mean_estimates)           

    def get_bootstrap_mean_diffs(self,estimate_data:tuple[pd.DataFrame], mean_data:tuple[pd.DataFrame])->tuple[pd.DataFrame]:
        means_diff = []

        for ed, md  in zip(estimate_data, mean_data):
            bootstrap_mean_diffs = ed[self.estimate_keys].copy()

            for dm in self.desired_results_metrics:   
                for index, row in ed.iterrows():

                        filtered_mean = md[np.logical_and.reduce([md[k] == row[k] for k in self.bootstrap_keys])][dm].item()
                        bootstrap_mean_diffs.loc[index,f'{dm}'] = row[dm] - filtered_mean

            self.save_csv(bootstrap_mean_diffs, 'bootstrap_mean_diffs')
            means_diff.append(bootstrap_mean_diffs)
        return tuple(means_diff)

    def get_standard_error(self, mean_diff: np.array) -> float:
        B = len(mean_diff)
        if B < 2:
            return np.nan  # undefined with fewer than 2 bootstrap samples
        return np.sqrt((mean_diff**2).sum() / (B - 1))

    def get_standard_bias(self, mean_diff: np.array)-> float:
        return np.sqrt((mean_diff**2).sum() / len(mean_diff))

    def get_bootstrap_standard_error(self,mean_diff_data: tuple[pd.DataFrame])->tuple[pd.DataFrame]:
            for mdd in mean_diff_data:
                bootstrap_std_error = pd.DataFrame([])
                for dm in self.desired_results_metrics:
                    standard_errors = mdd.groupby(self.bootstrap_keys)[dm].apply(lambda x: self.get_standard_error(x)).reset_index()

                    missing_columns = standard_errors.columns.difference(bootstrap_std_error.columns)

                    for mc in missing_columns:
                        bootstrap_std_error = pd.concat([bootstrap_std_error ,standard_errors[mc]], axis=1)
                self.save_csv(bootstrap_std_error, 'bootstrap_standard_errors')

    def get_bootstrap_standard_bias(self,mean_diff_data: tuple[pd.DataFrame])->tuple[pd.DataFrame]:
            for mdd in mean_diff_data:
                bootstrap_std_bias = pd.DataFrame([])
                for dm in self.desired_results_metrics:
                    standard_errors = mdd.groupby(self.bootstrap_keys)[dm].apply(lambda x: self.get_standard_bias(x)).reset_index()

                    missing_columns = standard_errors.columns.difference(bootstrap_std_bias.columns)

                    for mc in missing_columns:
                        bootstrap_std_bias = pd.concat([bootstrap_std_bias ,standard_errors[mc]], axis=1)
                self.save_csv(bootstrap_std_bias, 'bootstrap_standard_bias')

    def get_confidence_interval(self, estimates:np.array)-> tuple[float]:
        t_crit = t.ppf(1-self.significance_level/2, df = len(estimates))
        norm = np.sqrt(np.var(estimates) / len(estimates))
        l_bound = float(estimates.mean() - t_crit *  norm)
        u_bound = float(estimates.mean() + t_crit *  norm)

        return l_bound, u_bound
    
    def get_bootstrap_confidence_intervals(self, mean_bootstrap_estimates: tuple[pd.DataFrame])-> tuple[pd.DataFrame]:

        for mbe in mean_bootstrap_estimates:
            bootstrap_confidence_intervals = pd.DataFrame([])
            for dm in self.desired_results_metrics:
                bootstrap_confidence_interval = mbe.groupby(self.bootstrap_keys)[dm].apply(lambda x: (self.get_confidence_interval(x))).reset_index()

                missing_columns = bootstrap_confidence_interval.columns.difference(bootstrap_confidence_intervals.columns)

                for mc in missing_columns:
                    bootstrap_confidence_intervals = pd.concat([bootstrap_confidence_intervals,bootstrap_confidence_interval[mc]], axis=1)
            self.save_csv(bootstrap_confidence_intervals, 'bootstrap_ci')

    def get_median_bootstrap_estimates(self, estimate_data:tuple[pd.DataFrame] ) -> tuple[pd.DataFrame]:
        # Same as get_mean_bootstrap_estimates, but the median of the bootstrap estimates per bootstrap key
        bootstrap_median_estimates = []
        for ed in estimate_data:
            bootstrap_medians = pd.DataFrame([])
            bootstrap_group = ed.groupby(self.bootstrap_keys)

            for dm in self.desired_results_metrics:

                bootstrap_median = bootstrap_group[dm].median().reset_index()
                missing_columns = bootstrap_median.columns.difference(bootstrap_medians.columns)

                for mc in missing_columns:
                    bootstrap_medians = pd.concat([bootstrap_medians, bootstrap_median[mc]], axis=1)

            self.save_csv(bootstrap_medians, 'median_bootstrap_estimates')

            bootstrap_median_estimates.append(bootstrap_medians)
        return tuple(bootstrap_median_estimates)

    def get_bootstrap_median_diffs(self, estimate_data:tuple[pd.DataFrame], median_data:tuple[pd.DataFrame])->tuple[pd.DataFrame]:
        # Same as get_bootstrap_mean_diffs, but each bootstrap estimate minus the median: theta_i - median
        # (abs() of these is what Levene's test with the median centre / Brown-Forsythe compares between groups)
        medians_diff = []

        for ed, md in zip(estimate_data, median_data):
            bootstrap_median_diffs = ed[self.estimate_keys].copy()

            for dm in self.desired_results_metrics:
                for index, row in ed.iterrows():

                        # the median row with the same bootstrap keys (model, phases, t[, event]) as this estimate
                        filtered_median = md[np.logical_and.reduce([md[k] == row[k] for k in self.bootstrap_keys])][dm].item()
                        bootstrap_median_diffs.loc[index,f'{dm}'] = row[dm] - filtered_median

            self.save_csv(bootstrap_median_diffs, 'bootstrap_median_diffs')
            medians_diff.append(bootstrap_median_diffs)
        return tuple(medians_diff)



    def paired_groups(self, baseline_df: pd.DataFrame, champion_df: pd.DataFrame):
        # Every (champion, baseline) pair, matched on every bootstrap key except the model: (phases, t[, event]).
        # Yields (champion, baseline, {match key: value}, champion rows, baseline rows).
        match_keys = [k for k in self.bootstrap_keys if k != 'model_type']
        base_groups = dict(list(baseline_df.groupby(['model_type'] + match_keys)))
        baselines = baseline_df['model_type'].unique()
        for keys, champ in champion_df.groupby(['model_type'] + match_keys):
            model, group = keys[0], keys[1:]
            for bl in baselines:
                base = base_groups.get((bl, *group))
                if base is not None:
                    yield model, bl, dict(zip(match_keys, group)), champ, base

    def mean_t(self, y_base: np.ndarray, y_champ: np.ndarray, best: str) -> dict[str, float]:
        # One-sided Welch t-test (no equal-variance assumption) that the champion's mean is better.
        #   best 'min':  HA: mean(champion) < mean(baseline)
        #   best 'zero': HA: mean(|champion|) < mean(|baseline|)   (distance of the lag from 0)
        #   best 'max':  HA: mean(champion) > mean(baseline)
        # t = (better side) / sqrt(s_b^2/n_b + s_c^2/n_c), so a large positive t favours the champion.
        # df = (s_b^2/n_b + s_c^2/n_c)^2 / ((s_b^2/n_b)^2/(n_b-1) + (s_c^2/n_c)^2/(n_c-1)), p = P(T_df >= t)
        if best == 'zero':
            y_base, y_champ = np.abs(y_base), np.abs(y_champ)
        n1, n2 = len(y_base), len(y_champ)
        out = {'n_baseline': n1, 'n_champion': n2, 'mean_baseline': np.nan, 'mean_champion': np.nan,
               'std_err': np.nan, 't_stat': np.nan, 'df': np.nan, 'p_value': np.nan}
        if n1 < 2 or n2 < 2:
            return out
        m1, m2 = y_base.mean(), y_champ.mean()
        v1, v2 = y_base.var(ddof=1) / n1, y_champ.var(ddof=1) / n2
        se = np.sqrt(v1 + v2)
        out.update(mean_baseline=m1, mean_champion=m2, std_err=se)
        if se == 0:
            return out
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
        t_stat = (m2 - m1) / se if best == 'max' else (m1 - m2) / se
        out.update(t_stat=t_stat, df=df, p_value=t.sf(t_stat, df))
        return out

    def perform_mean_test(self, mean_estimates: tuple[pd.DataFrame]) -> pd.DataFrame:
        # Is the champion's mean better than the baseline's? Direction per metric from self.metric_info['best'].
        # mean_estimates = (baseline, champion) bootstrap estimates from get_mean_estimates.
        baseline_est, champion_est = mean_estimates
        hypotheses = {'min':  'mean(champion) < mean(baseline)',
                      'zero': 'mean(|champion|) < mean(|baseline|)',
                      'max':  'mean(champion) > mean(baseline)'}
        rows = []
        for model, bl, group, champ, base in self.paired_groups(baseline_est, champion_est):
            for dm in self.desired_results_metrics:
                best = self.metric_info[dm]['best']
                res = self.mean_t(base[dm].dropna().to_numpy(), champ[dm].dropna().to_numpy(), best)
                rows.append({'model_type': model, 'baseline': bl, **group, 'metric': dm,
                             'h_a': hypotheses[best], **res})

        mean_test = pd.DataFrame(rows)
        mean_test['reject_h0'] = mean_test['p_value'] < self.significance_level
        self.save_test_csv(mean_test, 'mean_test')   # -> metrics_analysis/mean_test_<mode>.csv
        return mean_test


# ---------- pipeline stages (called from main.py: analyze / plot) ----------

def run_bootstrap(a: Analysis, bootstrap_data: tuple[pd.DataFrame]) -> tuple[pd.DataFrame]:
    # Mean, SE, bias and CI of the bootstrap estimates, for whichever mode is set
    mean_estimates = a.get_mean_estimates(bootstrap_data)
    mean_of_bootstrap_estimates = a.get_mean_bootstrap_estimates(mean_estimates)
    diffs = a.get_bootstrap_mean_diffs(mean_estimates, mean_of_bootstrap_estimates)

    std_err = a.get_bootstrap_standard_error(diffs)
    # a.get_bootstrap_standard_bias(diffs)
    # a.get_bootstrap_confidence_intervals(mean_estimates)

    # Median counterparts: median of the bootstrap estimates and each estimate's distance from it
    median_of_bootstrap_estimates = a.get_median_bootstrap_estimates(mean_estimates)
    median_diffs = a.get_bootstrap_median_diffs(mean_estimates, median_of_bootstrap_estimates)

    # Hypothesis test: is each champion's mean better than each baseline's? Only on the per-dataset values;
    # per event there are too few baseline samples (1-4) for a meaningful test.
    if a.bootstrap_mode == 'event_avg':
        a.perform_mean_test(mean_estimates)

    return mean_estimates


def run_results_analysis(a: Analysis) -> None:
    # Bootstrap CSVs from every results.txt, twice:
    #   per dataset (the "Average ..." lines) -> results/metrics_analysis/csv/<baseline|champion>/event_avg/
    #   per event   (the "Event i" blocks)    -> results/metrics_analysis/csv/<baseline|champion>/events/
    a.set_bootstrap_mode('event_avg')
    run_bootstrap(a, a.compile_bootstrap_data('results'))
    a.set_bootstrap_mode('events')
    run_bootstrap(a, a.compile_event_bootstrap_data())
    a.set_bootstrap_mode('event_avg')
    # Train Time per run -> results/metrics_analysis/csv/train_time.csv
    a.get_train_times()


def build_prediction_database(a: Analysis) -> tuple[dict]:
    # Per-event database from every predictions.txt: bb[i][name][mt][t][p][ds][e]
    bootstrap_data_predictions = a.compile_bootstrap_data('predictions')
    bb = a.get_predictions_by_dataset(bootstrap_data_predictions)

    # mean and ssx of predictions
    bb = a.extend_prediction_database(bb, 'mean', a.get_prediction_mean, 'data')
    bb = a.extend_prediction_database(bb, 'ssx', a.get_prediction_ssx, 'data', 'mean')

    # regression of true protons on predictions
    bb = a.extend_prediction_database(bb, 'b1', a.get_prediction_b1, 'data', 'mean', 'ssx', 'true')
    bb = a.extend_prediction_database(bb, 'b0', a.get_prediction_b0, 'mean', 'b1', 'true')
    bb = a.extend_prediction_database(bb, 'sse', a.get_prediction_sse, 'data', 'b0', 'b1', 'true')
    bb = a.extend_prediction_database(bb, 's2', a.get_prediction_sample_variance, 'data', 'sse')
    bb = a.extend_prediction_database(bb, 'b1_std_err', a.get_prediction_b1_std_err, 's2', 'ssx')
    bb = a.extend_prediction_database(bb, 't_crit', a.get_prediction_t_critical_values, 'data')
    bb = a.extend_prediction_database(bb, 'ci', a.get_prediction_confidence_intervals, 'b1', 'b1_std_err', 't_crit')
    bb = a.extend_prediction_database(bb, 'b0_std_err', a.get_prediction_b0_std_err, 'data', 'mean', 'ssx', 's2')
    bb = a.extend_prediction_database(bb, 'ci_b0', a.get_prediction_confidence_intervals, 'b0', 'b0_std_err', 't_crit')

    # R^2 of the predictions themselves: 1 - sum((true - pred)^2) / sum((true - mean(true))^2)
    bb = a.extend_prediction_database(bb, 'ssr', a.get_prediction_ssr, 'data', 'true')
    bb = a.extend_prediction_database(bb, 'sst', a.get_prediction_sst, 'true')
    bb = a.extend_prediction_database(bb, 'r2', a.get_prediction_r2, 'ssr', 'sst')

    # PI is not stored: PlotStuff computes it from the bootstrap-mean parameters for what it plots.
    return bb


def prediction_database_path(a: Analysis) -> Path:
    return a.results_folder / 'true_vs_pred' / 'prediction_database.pkl'


def save_prediction_database(a: Analysis, bb: tuple[dict]) -> None:
    # Handoff from the analyze stage to the plot stage
    path = prediction_database_path(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(bb, f)


def load_prediction_database(a: Analysis) -> tuple[dict]:
    with open(prediction_database_path(a), 'rb') as f:
        return pickle.load(f)


def load_mean_estimates(a: Analysis) -> tuple[pd.DataFrame]:
    # The estimates written by get_mean_estimates for the current bootstrap mode, for the histogram plots
    folder = a.results_folder / a.metrics_folder
    return tuple(pd.read_csv(a.csv_path(group, 'mean_estimates')) for group in ('baseline', 'champion'))


def run_plots(a: Analysis, plots: PlotStuff, bb: tuple[dict]) -> None:
    # Histograms per dataset -> results/metrics_analysis/hists/<phases>/<metric>/events_avg.png
    a.set_bootstrap_mode('event_avg')
    plots.plot_mean_estimates(load_mean_estimates(a))
    # Histograms per event + best-model counts -> results/metrics_analysis/hists/<phases>/<metric>/
    a.set_bootstrap_mode('events')
    plots.plot_event_metric_histograms(load_mean_estimates(a))
    a.set_bootstrap_mode('event_avg')
    # Train time per model -> results/metrics_analysis/hists/<phases>/training_time.png
    plots.plot_train_time(pd.read_csv(a.results_folder / a.metrics_folder / 'train_time.csv'))
    # Reject / don't-reject grid per champion -> results/metrics_analysis/hypothesis_grids/<champion>.png
    plots.plot_hypothesis_grid(pd.read_csv(a.results_folder / 'metrics_analysis' / 'mean_test_event_avg.csv'))
    # Per-event true vs pred + best-model counts -> results/true_vs_pred/
    plots.plot_true_vs_pred_events(bb)

    # TODO: rebuild for the per-event database (they still expect bb[...][ds] arrays, not bb[...][ds][e])
    # a.save_true_vs_pred_stats(bb)
    # plots.plot_true_vs_pred(bb)
    # plots.plot_true_vs_pred_all_models(bb)


if __name__ == "__main__":
    # Running this file directly does analyze + plot (same as: python main.py analyze, then python main.py plot --show)
    a = Analysis()

    run_results_analysis(a)
    bb = build_prediction_database(a)
    save_prediction_database(a, bb)
    run_plots(a, PlotStuff(a, show=True), bb)
