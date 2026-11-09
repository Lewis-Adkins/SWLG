from pathlib import Path
import pickle
import shutil
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
        self.desired_results_metrics       = ['MAE', 'MSE', 'O2P lag', 'O2T lag', 'ln10 lag', 'PE', 'F1']
        self.phases                = ['phases', 'nophases']
        self.file_type             = ['results', 'predictions']
        self.times                 = ['t+6', 't+12']
        self.train_split           = train_split
        self.true_protons          = self.get_true_protons()
        self.significance_level    = significance_level
        self.results_metrics       = self.desired_results_metrics                    # "Average ..." lines of results.txt
        self.event_metrics         = ['MAE', 'MSE', 'O2P lag', 'O2T lag', 'ln10 lag', 'PE', 'F1']  # "Event i" blocks (r2 = PE per event)
        # The one metric table: what "best" means (lowest, highest, closest to 0) and each metric's plot folder.
        self.metric_info = {
            'MAE':        {'best': 'min',  'folder': 'mae'},
            'MSE':        {'best': 'min',  'folder': 'mse'},
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

    def save_test_csv(self, test_table: pd.DataFrame, test_name: str) -> None:
        # Hypothesis test tables (all champion x baseline pairs in one file):
        # results/metrics_analysis/<name>_<event_avg|events>.csv. %.6g so small p-values are not rounded to 0.
        test_path = self.results_folder / 'metrics_analysis' / f'{test_name}_{self.bootstrap_mode}.csv'
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_table.to_csv(test_path, index=False, float_format='%.6g')

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


    def get_mean_bootstrap_estimates(self,estimate_data:tuple[pd.DataFrame], save: bool = True):
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



            if save:
                self.save_csv(bootstrap_means, 'mean_bootstrap_estimates')

            bootstrap_mean_estimates.append(bootstrap_means)
        return tuple(bootstrap_mean_estimates)           

    def get_bootstrap_mean_diffs(self,estimate_data:tuple[pd.DataFrame], mean_data:tuple[pd.DataFrame], save: bool = True)->tuple[pd.DataFrame]:
        means_diff = []

        for ed, md  in zip(estimate_data, mean_data):
            bootstrap_mean_diffs = ed[self.estimate_keys].copy()

            for dm in self.desired_results_metrics:   
                for index, row in ed.iterrows():

                        filtered_mean = md[np.logical_and.reduce([md[k] == row[k] for k in self.bootstrap_keys])][dm].item()
                        bootstrap_mean_diffs.loc[index,f'{dm}'] = row[dm] - filtered_mean

            if save:
                self.save_csv(bootstrap_mean_diffs, 'bootstrap_mean_diffs')
            means_diff.append(bootstrap_mean_diffs)
        return tuple(means_diff)

    def get_standard_error(self, mean_diff: np.array) -> float:
        mean_diff = np.asarray(mean_diff, dtype=float)
        mean_diff = mean_diff[~np.isnan(mean_diff)]   # missing estimates (e.g. an undefined PE) don't count in B
        B = len(mean_diff)
        if B < 2:
            return np.nan  # undefined with fewer than 2 bootstrap samples
        return np.sqrt((mean_diff**2).sum() / (B - 1))

    def get_standard_bias(self, mean_diff: np.array)-> float:
        return np.sqrt((mean_diff**2).sum() / len(mean_diff))

    def get_bootstrap_standard_error(self,mean_diff_data: tuple[pd.DataFrame], save: bool = True)->tuple[pd.DataFrame]:
            std_errors = []
            for mdd in mean_diff_data:
                bootstrap_std_error = pd.DataFrame([])
                for dm in self.desired_results_metrics:
                    standard_errors = mdd.groupby(self.bootstrap_keys)[dm].apply(lambda x: self.get_standard_error(x)).reset_index()

                    missing_columns = standard_errors.columns.difference(bootstrap_std_error.columns)

                    for mc in missing_columns:
                        bootstrap_std_error = pd.concat([bootstrap_std_error ,standard_errors[mc]], axis=1)
                if save:
                    self.save_csv(bootstrap_std_error, 'bootstrap_standard_errors')
                std_errors.append(bootstrap_std_error)
            return tuple(std_errors)

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



    def champion_baseline_pairs(self, baseline_table: pd.DataFrame, champion_table: pd.DataFrame):
        # Every (champion, baseline) pair, matched on every bootstrap key except the model: (phases, t[, event]).
        # Yields (champion name, baseline name, {match key: value}, champion rows, baseline rows).
        match_keys = [key for key in self.bootstrap_keys if key != 'model_type']
        baseline_rows_by_group = dict(list(baseline_table.groupby(['model_type'] + match_keys)))
        baseline_names = baseline_table['model_type'].unique()
        for group_key, champion_rows in champion_table.groupby(['model_type'] + match_keys):
            champion_name, match_values = group_key[0], group_key[1:]
            for baseline_name in baseline_names:
                baseline_rows = baseline_rows_by_group.get((baseline_name, *match_values))
                if baseline_rows is not None:
                    yield champion_name, baseline_name, dict(zip(match_keys, match_values)), champion_rows, baseline_rows

    def welch_t_stat(self, baseline_mean: float, baseline_std_error: float, baseline_n: int,
                     champion_mean: float, champion_std_error: float, champion_n: int,
                     best_direction: str) -> dict[str, float]:

        test_result = {'n_baseline': baseline_n, 'n_champion': champion_n,
                       'mean_baseline': baseline_mean, 'mean_champion': champion_mean,
                       'std_error_of_difference': np.nan, 't_stat': np.nan, 'df': np.nan, 'p_value': np.nan}
        if baseline_n < 2 or champion_n < 2 or np.isnan(baseline_std_error) or np.isnan(champion_std_error):
            return test_result
        
        baseline_mean_variance = baseline_std_error ** 2 / baseline_n    # s_b^2 / n_b
        champion_mean_variance = champion_std_error ** 2 / champion_n    # s_c^2 / n_c
        std_error_of_difference = np.sqrt(baseline_mean_variance + champion_mean_variance)
        test_result['std_error_of_difference'] = std_error_of_difference

        if std_error_of_difference == 0:
            return test_result
        
        degrees_of_freedom = (baseline_mean_variance + champion_mean_variance) ** 2 / (
            baseline_mean_variance ** 2 / (baseline_n - 1) + champion_mean_variance ** 2 / (champion_n - 1))
        
        if best_direction == 'max':
            t_stat = (champion_mean - baseline_mean) / std_error_of_difference
        else:
            t_stat = (baseline_mean - champion_mean) / std_error_of_difference

        test_result.update(t_stat=t_stat, df=degrees_of_freedom, p_value=t.sf(t_stat, degrees_of_freedom))
        
        return test_result

    def perform_welch_test(self, mean_estimates: tuple[pd.DataFrame], mean_of_bootstrap_estimates: tuple[pd.DataFrame],
                           std_err: tuple[pd.DataFrame]) -> pd.DataFrame:

        lag_metrics = [metric for metric in self.desired_results_metrics if self.metric_info[metric]['best'] == 'zero']

        abs_lag_estimates = tuple(estimates.assign(**{metric: estimates[metric].abs() for metric in lag_metrics})
                                  for estimates in mean_estimates)
        
        abs_lag_bootstrap_means = self.get_mean_bootstrap_estimates(abs_lag_estimates, save=False)

        abs_lag_std_errors = self.get_bootstrap_standard_error(
            self.get_bootstrap_mean_diffs(abs_lag_estimates, abs_lag_bootstrap_means, save=False), save=False)

        # One summary row per bootstrap group: mean, standard error and B for every metric (|lag| for the lags).
        # B = estimates that aren't missing, the same ones get_standard_error uses.
        group_summaries = []

        for estimates, bootstrap_means, std_errors, lag_bootstrap_means, lag_std_errors in zip(
                mean_estimates, mean_of_bootstrap_estimates, std_err, abs_lag_bootstrap_means, abs_lag_std_errors):
            
            summary_means = bootstrap_means.set_index(self.bootstrap_keys)
            summary_means[lag_metrics] = lag_bootstrap_means.set_index(self.bootstrap_keys)[lag_metrics]
            summary_std_errors = std_errors.set_index(self.bootstrap_keys)
            summary_std_errors[lag_metrics] = lag_std_errors.set_index(self.bootstrap_keys)[lag_metrics]
            estimate_counts = estimates.groupby(self.bootstrap_keys)[self.desired_results_metrics].count()

            group_summaries.append(summary_means.join(summary_std_errors, rsuffix='_se')
                                   .join(estimate_counts, rsuffix='_B').reset_index())
            
        baseline_summary, champion_summary = group_summaries

        alternative_hypotheses = {'min':  'mean(champion) < mean(baseline)',
                                  'zero': 'mean(|champion|) < mean(|baseline|)',
                                  'max':  'mean(champion) > mean(baseline)'}
        test_rows = []

        for champion_name, baseline_name, match_values, champion_rows, baseline_rows in \
                self.champion_baseline_pairs(baseline_summary, champion_summary):
            
            baseline_row, champion_row = baseline_rows.iloc[0], champion_rows.iloc[0]

            for metric in self.desired_results_metrics:
                best_direction = self.metric_info[metric]['best']
                test_result = self.welch_t_stat(
                    baseline_row[metric], baseline_row[f'{metric}_se'], int(baseline_row[f'{metric}_B']),
                    champion_row[metric], champion_row[f'{metric}_se'], int(champion_row[f'{metric}_B']),
                    best_direction)
                
                test_rows.append({'model_type': champion_name, 'baseline': baseline_name, **match_values,
                                  'metric': metric, 'h_a': alternative_hypotheses[best_direction], **test_result})

        welch_test = pd.DataFrame(test_rows)
        welch_test['reject_h0'] = welch_test['p_value'] < self.significance_level
        self.save_test_csv(welch_test, 'welch_test')   # -> metrics_analysis/welch_test_<mode>.csv
        return welch_test

    def get_levene_samples(self, median_diff_data: tuple[pd.DataFrame]) -> tuple[pd.DataFrame]:
        # Levene's test steps 1-4 (lecture notes p.195), from get_bootstrap_median_diffs (steps 1-2: theta_i - median):
        #   (3) take |theta_i - median|
        #   (4) for a group with an odd number of estimates, remove one zero (the median itself)
        # Same rows as median_diff_data; the removed value becomes NaN.
        levene_samples = []
        for median_diffs in median_diff_data:
            abs_median_diffs = median_diffs.copy()
            for metric in self.desired_results_metrics:
                abs_median_diffs[metric] = abs_median_diffs[metric].abs()
                for _, group_abs_diffs in abs_median_diffs.groupby(self.bootstrap_keys)[metric]:
                    group_abs_diffs = group_abs_diffs.dropna()
                    if len(group_abs_diffs) % 2 != 0:
                        # the smallest |theta_i - median| is the zero
                        abs_median_diffs.loc[group_abs_diffs.idxmin(), metric] = np.nan
            levene_samples.append(abs_median_diffs)
        return tuple(levene_samples)

    def levene_pooled_t_stat(self, baseline_abs_diffs: np.ndarray, champion_abs_diffs: np.ndarray) -> dict[str, float]:
        # Levene's test step 5 (lecture notes p.195): pooled two-sample t-test on the |theta_i - median| samples, one-sided.
        #   H0: var(champion) >= var(baseline)   vs   HA: var(champion) < var(baseline)
        # A smaller variance means a smaller mean |theta_i - median|, so HA is mean(champion) < mean(baseline).
        #   s_p^2 = ((n_b - 1) s_b^2 + (n_c - 1) s_c^2) / (n_b + n_c - 2),  s from get_standard_error
        #   t = (mean_b - mean_c) / (s_p * sqrt(1/n_b + 1/n_c)),  df = n_b + n_c - 2,  p = P(T_df >= t)
        baseline_n, champion_n = len(baseline_abs_diffs), len(champion_abs_diffs)
        test_result = {'n_baseline': baseline_n, 'n_champion': champion_n, 'mean_baseline': np.nan, 'mean_champion': np.nan,
                       'pooled_std': np.nan, 't_stat': np.nan, 'df': np.nan, 'p_value': np.nan}
        if baseline_n < 2 or champion_n < 2:
            return test_result   # no variance estimate for a group this small
        baseline_mean, champion_mean = baseline_abs_diffs.mean(), champion_abs_diffs.mean()
        baseline_std_error = self.get_standard_error(baseline_abs_diffs - baseline_mean)
        champion_std_error = self.get_standard_error(champion_abs_diffs - champion_mean)
        degrees_of_freedom = baseline_n + champion_n - 2
        pooled_std = np.sqrt(((baseline_n - 1) * baseline_std_error ** 2
                              + (champion_n - 1) * champion_std_error ** 2) / degrees_of_freedom)
        test_result.update(mean_baseline=baseline_mean, mean_champion=champion_mean, pooled_std=pooled_std,
                           df=degrees_of_freedom)
        if pooled_std == 0:
            return test_result
        t_stat = (baseline_mean - champion_mean) / (pooled_std * np.sqrt(1 / baseline_n + 1 / champion_n))
        test_result.update(t_stat=t_stat, p_value=t.sf(t_stat, degrees_of_freedom))
        return test_result

    def perform_levene_test(self, median_diff_data: tuple[pd.DataFrame]) -> pd.DataFrame:
        # Is the champion's spread smaller than the baseline's? HA: var(champion) < var(baseline), for every
        # champion x baseline pair, metric and bootstrap group (raw values, also for the lags: this is about spread).
        # median_diff_data = (baseline, champion) from get_bootstrap_median_diffs.
        baseline_levene_samples, champion_levene_samples = self.get_levene_samples(median_diff_data)
        test_rows = []
        for champion_name, baseline_name, match_values, champion_rows, baseline_rows in \
                self.champion_baseline_pairs(baseline_levene_samples, champion_levene_samples):
            for metric in self.desired_results_metrics:
                test_result = self.levene_pooled_t_stat(baseline_rows[metric].dropna().to_numpy(),
                                                        champion_rows[metric].dropna().to_numpy())
                test_rows.append({'model_type': champion_name, 'baseline': baseline_name, **match_values,
                                  'metric': metric, 'h_a': 'var(champion) < var(baseline)', **test_result})

        levene_test = pd.DataFrame(test_rows)
        levene_test['reject_h0'] = levene_test['p_value'] < self.significance_level
        self.save_test_csv(levene_test, 'levene_test')   # -> metrics_analysis/levene_test_<mode>.csv
        return levene_test


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
        a.perform_welch_test(mean_estimates, mean_of_bootstrap_estimates, std_err)
        a.perform_levene_test(median_diffs)

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


# ---------- F1 per alerting approach (Torres's W / EW / EAW / AW) ----------
# Port of lib/torres/legacy/time_series_classification.py, but on each dataset's own test targets (row positions
# within the test set, events found by timestamp) instead of data_0's absolute rows, so it works for every
# bootstrap dataset. An alert starts when the prediction reaches ln(10) and ends alert_window steps after the
# last prediction above it. The approaches differ in two switches:
#   extend: keep extending the alert while another prediction >= ln(10) comes within alert_window (EW, EAW)
#           instead of only while the predictions stay above ln(10) step after step (W, AW)
#   early:  the alert starts when the forecast is issued (target time - t) instead of at the target time (EAW, AW)
# TP = event whose threshold crossing falls inside an alert, FN = event with none, FP = alert that touches no event.
F1_APPROACHES = {'W': (False, False), 'EW': (True, False), 'EAW': (True, True), 'AW': (False, True)}
ALERT_WINDOWS = [1, 3, 6, 9, 12, 24, 72]   # 5-min steps: 5 min to 6 h


def alert_periods(predictions: np.ndarray, prediction_time: int, extend: bool, early: bool,
                  alert_window: int) -> np.ndarray:
    # (start, end) test positions of every alert, following _count_tp_fp_fn in the legacy code
    above = predictions >= np.log(10)
    n = len(predictions)
    above_positions = np.flatnonzero(above)
    alerts = []
    t = above_positions[0] if len(above_positions) else n
    while t < n:
        start = t - prediction_time if early else t
        if extend:
            window = above[t + 1: t + alert_window + 1]
            while window.any():
                t += len(window) - int(np.argmax(window[::-1]))   # jump to the last prediction above ln(10)
                window = above[t + 1: t + alert_window + 1]
        else:
            while t + 1 < n and above[t + 1]:
                t += 1
        alerts.append((start, t + alert_window))
        t += alert_window + (prediction_time if early else 0)
        # skip ahead to the next prediction above ln(10)
        later = np.searchsorted(above_positions, t)
        t = above_positions[later] if later < len(above_positions) else n
    return np.array(alerts, dtype=float).reshape(-1, 2)


def alert_f1(alerts: np.ndarray, events: np.ndarray) -> tuple[int, int, int, float]:
    # TP / FP / FN / F1 of the alerts against the events, each an (n, 2) array of (start, end) test positions
    # (events: threshold crossing, event end)
    if len(events) == 0:
        return 0, len(alerts), 0, 0.0
    a0, a1 = alerts[:, :1], alerts[:, 1:]
    e0, e1 = events[:, 0][None, :], events[:, 1][None, :]
    catches_start = (a0 < e0) & (e0 < a1)                     # alert covers the threshold crossing
    inside = (e0 < a0) & (a1 < e1)                            # alert entirely inside the event
    covers_end = (e0 < a0) & (a0 < e1) & (e1 < a1)            # alert overlaps the end only
    tp = int(catches_start.any(axis=0).sum())
    fn = len(events) - tp
    fp = int((~(catches_start | inside | covers_end).any(axis=1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return tp, fp, fn, f1


def compute_f1_approaches(a: Analysis, data_folder: str, event_timestamps: list[list[str]]) -> pd.DataFrame:
    # F1 of every run's saved predictions.txt for each approach and alert window.
    # One row per (model_type, phases, dataset, t, run, approach, alert_window) -> metrics_analysis/csv/f1_approaches.csv
    rows = []
    prediction_files = sorted(a.results_folder.rglob('predictions.txt'))
    for dataset_name in sorted({parse_results_path(a.results_folder, f)['dataset'] for f in prediction_files}):
        data = pd.read_csv(Path(data_folder) / f"data_{dataset_name.split('_')[1]}.csv", usecols=['time'])
        for prediction_time in (6, 12):
            n_test = n_test_samples(len(data), prediction_time, a.train_split)
            test_times = data['time'].iloc[len(data) - n_test:].tolist()
            position = {time: i for i, time in enumerate(test_times)}
            events = np.array([(position[threshold], position.get(end, n_test - 1))
                               for _, threshold, _, end in event_timestamps if threshold in position], dtype=float)
            events = events.reshape(-1, 2)
            for f in prediction_files:
                info = parse_results_path(a.results_folder, f)
                if info['dataset'] != dataset_name or info['t'] != f't+{prediction_time}':
                    continue
                predictions = np.loadtxt(f)
                if len(predictions) != n_test:
                    print(f"  SKIPPED {f}: {len(predictions)} predictions for {n_test} test targets")
                    continue
                for approach, (extend, early) in F1_APPROACHES.items():
                    for alert_window in ALERT_WINDOWS:
                        tp, fp, fn, f1 = alert_f1(alert_periods(predictions, prediction_time, extend, early,
                                                                alert_window), events)
                        rows.append(dict(model_type=info['model_type'], phases=info['phases'], dataset=dataset_name,
                                         t=info['t'], run=info.get('model_name', 'run'), approach=approach,
                                         alert_window=alert_window, TP=tp, FP=fp, FN=fn, F1=f1))
        print(f"  F1 approaches: {dataset_name} done")
    table = pd.DataFrame(rows)
    out = a.results_folder / a.metrics_folder / 'f1_approaches.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format='%.4f')
    return table


# ---------- RQ2: warning time per event (causal alert rule) ----------
# An operator never knows the crossing time, so alerts are built only from the forecasts: Torres's EAW alert
# (alert_periods with extend=True, early=True) starts at the ISSUE time of the first forecast >= 10 pfu and stays on
# while another forecast >= 10 pfu comes within RQ2_ALERT_WINDOW steps. Scored afterwards against the true crossing c:
#   hit     = an alert is on at c (inclusive, so an alert starting exactly at c counts, e.g. persistence)
#   warning = (c - alert start) x 5 min; a missed event gets warning 0
RQ2_ALERT_WINDOW = 12   # 1 h


def compute_warning_times(a: Analysis, data_folder: str, event_timestamps: list[list[str]],
                          models: list[str] | None = None, check_alignment: bool = True,
                          save: bool = True) -> pd.DataFrame:
    # One row per (model_type, phases, dataset, t, run, event) -> metrics_analysis/csv/rq2_warning_times.csv
    # check_alignment: skip predictions.txt files older than their data_<k>.csv (the bootstrap data was regenerated
    # after those runs, so their test rows no longer line up with the events).
    rows, skipped = [], 0
    prediction_files = sorted(a.results_folder.rglob('predictions.txt'))
    cache = {}
    for f in prediction_files:
        info = parse_results_path(a.results_folder, f)
        if models is not None and info['model_type'] not in models:
            continue
        h = int(info['t'].split('+')[1])
        data_path = Path(data_folder) / f"data_{info['dataset'].split('_')[1]}.csv"
        if check_alignment and f.stat().st_mtime < data_path.stat().st_mtime:
            skipped += 1
            continue
        key = (info['dataset'], h)
        if key not in cache:
            times = pd.read_csv(data_path, usecols=['time'])['time']
            n_test = n_test_samples(len(times), h, a.train_split)
            position = {time: i for i, time in enumerate(times.iloc[len(times) - n_test:])}
            cache[key] = (n_test, [(i, position[thr]) for i, (_, thr, _, _) in enumerate(event_timestamps)
                                   if thr in position])
        n_test, events = cache[key]
        predictions = np.loadtxt(f)
        if len(predictions) != n_test or not events:
            continue
        alerts = alert_periods(predictions, h, extend=True, early=True, alert_window=RQ2_ALERT_WINDOW)
        for event, c in events:
            on = alerts[(alerts[:, 0] <= c) & (c <= alerts[:, 1])] if len(alerts) else alerts
            rows.append(dict(model_type=info['model_type'], phases=info['phases'], dataset=info['dataset'],
                             t=info['t'], run=info.get('model_name', 'run'), event=event + 1, hit=bool(len(on)),
                             warning_min=float((c - on[:, 0].min()) * 5) if len(on) else 0.0))
    if skipped:
        print(f"  RQ2 warning times: skipped {skipped} prediction files older than their data file (re-train them)")
    table = pd.DataFrame(rows)
    if save:
        out = a.results_folder / a.metrics_folder / 'rq2_warning_times.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
    return table


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
    # MAE / MSE box plots and bars of the bootstrap means (+- bootstrap standard error)
    # -> results/metrics_analysis/hists/<phases>/mae_mse_boxplot.png, mae_mse_bars.png
    plots.plot_mae_mse(load_mean_estimates(a),
                       tuple(pd.read_csv(a.csv_path(group, 'mean_bootstrap_estimates')) for group in ('baseline', 'champion')),
                       tuple(pd.read_csv(a.csv_path(group, 'bootstrap_standard_errors')) for group in ('baseline', 'champion')))
    # RQ1 summary figures in AGU print style (heatmap, box plots, volcano vs persistence)
    # -> SWLG-Paper/figs/results/rq1/rq1_<heatmap|box|volcano>_<phases>.pdf (+ .png previews, rq1_tests_<phases>.csv)
    paper_rq1 = Path(r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\rq1')
    for phases in a.phases:
        plots.plot_rq1_heatmap(load_mean_estimates(a), phases=phases, out_dir=paper_rq1)
        plots.plot_rq1_box(load_mean_estimates(a), phases=phases, out_dir=paper_rq1)
        plots.plot_rq1_volcano(load_mean_estimates(a), phases=phases, out_dir=paper_rq1)
    # PNG copies inside the repo so the Research Questions.md preview can show them (VS Code only loads images
    # from inside the workspace) -> results/metrics_analysis/rq1/
    repo_rq1 = a.results_folder / 'metrics_analysis' / 'rq1'
    repo_rq1.mkdir(parents=True, exist_ok=True)
    for png in paper_rq1.glob('rq1_*.png'):
        shutil.copy2(png, repo_rq1 / png.name)
    # RQ2: share of events warned >= R ahead vs assumed response time R
    # -> metrics_analysis/csv/rq2_warning_times.csv, SWLG-Paper/figs/results/rq2/rq2_warning_<phases>.pdf (+ .png),
    #    PNG copy in results/metrics_analysis/rq2/
    warning_times = compute_warning_times(a, 'data', read_event_timestamps(str(Path('data') / 'event_timestamps.txt')))
    paper_rq2 = Path(r'C:\Users\Lewis\Documents\SWLG-Paper\figs\results\rq2')
    repo_rq2 = a.results_folder / 'metrics_analysis' / 'rq2'
    repo_rq2.mkdir(parents=True, exist_ok=True)
    for phases in a.phases:
        if (warning_times['phases'] == phases).any():
            plots.plot_rq2_warning(warning_times, phases=phases, out_dir=paper_rq2)
    for png in paper_rq2.glob('rq2_*.png'):
        shutil.copy2(png, repo_rq2 / png.name)
    # Reject / don't-reject grids per champion -> results/metrics_analysis/hypothesis_grids/<welch|levene>/<champion>.png
    plots.plot_welch_grids(pd.read_csv(a.results_folder / 'metrics_analysis' / 'welch_test_event_avg.csv'))
    plots.plot_levene_grids(pd.read_csv(a.results_folder / 'metrics_analysis' / 'levene_test_event_avg.csv'))
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
