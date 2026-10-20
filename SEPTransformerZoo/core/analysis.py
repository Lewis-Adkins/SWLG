from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
class Analysis:
    def __init__(self):
        self.results_folder        = Path("results")
        self.model_names_w_results = self.get_model_names_w_results()
        self.baseline_names        = ['linearregression', 'posner', 'persistence']
        self.champion_names        = ['nn', 'rnn', 'rope', 'sin', 'zero', 'autogluon']
        self.desired_metrics       = ['Average MAE','Average O2P lag','Average O2T lag', 'Average ln10 lag', 'PE', 'Train Time']
        self.phases                = ['phases', 'nophases']
        self.file_type             = ['results', 'predictions']
        self.times                 = ['t+6', 't+12']
        
    def get_model_names_w_results(self)-> list[str]:
        return [mrf.name for mrf in self.results_folder.iterdir()]

    def get_pred_files_by_model(self,file_type: str)->pd.DataFrame:
        baseline_data = []
        champion_data = []

        for mn in self.model_names_w_results:
            for p in self.phases:
                for path in Path(f"results/{mn}/{p}").rglob("*.txt"):
                    path_parts = list(path.parts)
                    path_parts = path_parts + [str(path)]

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
        files = self.get_pred_files_by_model(file_type)
        if file_type == 'results':
                
            for f in files:
                for dm in self.desired_metrics:
                    
                    f[dm] = f['path'].apply(lambda p: self.pull_metrics_from_results_file(p,dm)) 


                if f['model_type'].iloc[0] in self.baseline_names:

                    f.to_csv("results/baseline_bootstrap.csv", index = True)
                else:
                    f.to_csv("results/champion_bootstrap.csv",  index = True)

        else:
            print("predictions not configured")


 
        return files


    def get_mean_estimates(self, data:tuple[pd.DataFrame])-> tuple[pd.DataFrame]:
        mean_estimates = []

        for d in data:
            bootstrap_means = pd.DataFrame([])
            bootstrap_group = d.groupby(['model_type', 'phases', 't','dataset'])
            for dm in self.desired_metrics:
                        
                bootstrap_mean = bootstrap_group[dm].mean().reset_index()
                missing_columns = bootstrap_mean.columns.difference(bootstrap_means.columns)

                for mc in missing_columns:
                    bootstrap_means = pd.concat([bootstrap_means, bootstrap_mean[mc]], axis=1)



            if bootstrap_means['model_type'].iloc[0] in self.baseline_names:
                bootstrap_means.to_csv('results/baseline_mean_estimates.csv', index=False)
            else:
                bootstrap_means.to_csv('results/champion_means_estimates.csv', index=False)

            mean_estimates.append(bootstrap_means)
        # print(tuple(means))
        return tuple(mean_estimates)


    def get_mean_bootstrap_estimates(self,estimate_data:tuple[pd.DataFrame]):
        bootstrap_mean_estimates = []
        for ed in estimate_data:
            bootstrap_means = pd.DataFrame([])
            bootstrap_group = ed.groupby(['model_type', 'phases', 't'])

            for dm in self.desired_metrics:
                        
                bootstrap_mean = bootstrap_group[dm].mean().reset_index()
                # print(bootstrap_mean)
                missing_columns = bootstrap_mean.columns.difference(bootstrap_means.columns)

                for mc in missing_columns:
                    bootstrap_means = pd.concat([bootstrap_means, bootstrap_mean[mc]], axis=1)



            if bootstrap_means['model_type'].iloc[0] in self.baseline_names:
                bootstrap_means.to_csv('results/baseline_mean_bootstrap_estimates.csv', index=False)
            else:
                bootstrap_means.to_csv('results/champion_mean_bootstrap_estimates.csv', index=False)

            bootstrap_mean_estimates.append(bootstrap_means)
        return tuple(bootstrap_mean_estimates)           

    def get_bootstrap_mean_diffs(self,estimate_data:tuple[pd.DataFrame], mean_data:tuple[pd.DataFrame])->tuple[pd.DataFrame]:
        means_diff = []

        for ed, md  in zip(estimate_data, mean_data):
            bootstrap_mean_diffs = ed[["model_type","phases","dataset","t"]].copy()
            if ed['model_type'].iloc[0] in self.baseline_names:
                path = 'results/baseline_bootstrap_mean_diffs.csv'
            else:
                path = 'results/champion_bootstrap_mean_diffs.csv'

            for dm in self.desired_metrics:   
                for index, row in ed.iterrows():

                        filtered_mean = md[(md['model_type'] == row['model_type']) & (md['t'] == row['t']) & (md['phases'] == row['phases'])][dm].item()
                        bootstrap_mean_diffs.loc[index,f'{dm}'] = row[dm] - filtered_mean

            bootstrap_mean_diffs.to_csv(path, index=False)
            means_diff.append(bootstrap_mean_diffs)
        return tuple(means_diff)

    def get_bootstrap_standard_error(self,mean_diff_data: tuple[pd.DataFrame])->tuple[pd.DataFrame]:
            for mdd in mean_diff_data:
                bootstrap_std_error = pd.DataFrame([])
                for dm in self.desired_metrics:
                    standard_errors = mdd.groupby(['model_type', 'phases', 't'])[dm].apply(lambda x: np.sqrt((x**2).sum() / (len(x)-1))).reset_index()

                    missing_columns = standard_errors.columns.difference(bootstrap_std_error.columns)

                    for mc in missing_columns:
                        bootstrap_std_error = pd.concat([bootstrap_std_error ,standard_errors[mc]], axis=1)
                if mdd['model_type'].iloc[0] in self.baseline_names:
                    path = 'results/baseline_bootstrap_standard_errors.csv'
                else:
                    path = 'results/champion_bootstrap_standard_errors.csv'

                bootstrap_std_error.to_csv(path, index=False)

    def get_bootstrap_standard_error(self,mean_diff_data: tuple[pd.DataFrame])->tuple[pd.DataFrame]:
            for mdd in mean_diff_data:
                bootstrap_std_error = pd.DataFrame([])
                for dm in self.desired_metrics:
                    standard_errors = mdd.groupby(['model_type', 'phases', 't'])[dm].apply(lambda x: np.sqrt((x**2).sum() / (len(x)-1))).reset_index()

                    missing_columns = standard_errors.columns.difference(bootstrap_std_error.columns)

                    for mc in missing_columns:
                        bootstrap_std_error = pd.concat([bootstrap_std_error ,standard_errors[mc]], axis=1)
                if mdd['model_type'].iloc[0] in self.baseline_names:
                    path = 'results/baseline_bootstrap_standard_errors.csv'
                else:
                    path = 'results/champion_bootstrap_standard_errors.csv'

                bootstrap_std_error.to_csv(path, index=False)

    def plot_diffs(self, mean_diff_data: tuple[pd.DataFrame])-> None:

        for mdd in mean_diff_data:
            for dm in self.desired_metrics:
                #  print(mdd.groupby(['model_type', 'phases', 't'])[dm]
                 plt.title(f'{mdd['model_type']}, {dm}')
                 plt.show()
                

a = Analysis()
bootstrap_data = a.compile_bootstrap_data('results')
base = bootstrap_data[0]
champ = bootstrap_data[1]

mean_estimates = a.get_mean_estimates(bootstrap_data)
mean_of_bootstrap_estimates = a.get_mean_bootstrap_estimates(mean_estimates)
diffs = a.get_bootstrap_mean_diffs(mean_estimates ,mean_of_bootstrap_estimates)

d = a.get_bootstrap_standard_error(diffs)

a.plot_diffs(diffs)