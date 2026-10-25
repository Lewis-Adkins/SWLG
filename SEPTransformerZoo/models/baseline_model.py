import numpy as np
from lib.torres.stats import *
from core.plotstuff import PlotStuff
from lib.utils.events import find_test_events


class BaselineModel:
    def __init__(self, config:dict):
        self.config = config

    def test_events(self, event_times: list[list[str]], targets_test: dict) -> list[list[str]]:
        # The events evaluate() scores: the ones inside this run's test targets (rule in lib/utils/events.py)
        return [event_times[i] for i in find_test_events(list(targets_test.keys()), event_times)]

    def slice_dictionary_keys(self,dictionary, start, end):
        """
        Slice a dictionary by keys rather than indices.
        :param dictionary: A dictionary object
        :param start: The starting key
        :param end: The ending key
        :return: The dictionary keys and values between the start and end keys
        """
        times = list(dictionary.keys())

        keys = times[times.index(start): times.index(end) + 1]
        return {key: dictionary[key] for key in keys}
    
    def train_test_split(self,x, y, train_ratio):
        """
        Split inputs and outputs into training and test sets.
        :param x: The inputs to the neural network
        :param y: The outputs corresponding to the given inputs
        :param train_ratio: The percentage of data to use for training
        :return: The training inputs and targets, and the test inputs and targets
        """
        train = x[:int(train_ratio * len(x))]
        targets_train = y[:int(train_ratio * len(y))]
        test = x[int(train_ratio * len(x)):]
        targets_test = y[int(train_ratio * len(y)):]
        return np.array(train), np.array(targets_train), np.array(test), np.array(targets_test)

    def pair_input_output(self,data, use_phase_inputs, prediction_time):
        """
        Pair inputs and outputs, and split into training and testing sets.
        :param data: A DataFrame object
        :param use_phase_inputs: Whether or not to use phase inputs
        :param prediction_time: The number of timesteps to predict ahead
        :return: Training and test inputs and targets
        """

        # Select features
        time = list(data['time'].values)
        electron = data['electron'].values
        electron_high = data['electron_high'].values
        proton = data['proton'].values
        phases = np.array([data['background'].values, data['rising'].values, data['falling'].values]).T

        # Pair inputs and outputs
        x = []
        y = []
        for t in range(24, len(data) - prediction_time):

            # Get current instance (past 2 hours plus current)
            x_curr = [electron[t - 24: t + 1], electron_high[t - 24: t + 1], proton[t - 24: t + 1]]

            # Adjust phase inputs by checking if there was a change in phase before the past 30 minutes
            # If so, then starting from last index, go backwards and replace up to t-6 with the value of t-6
            if use_phase_inputs:
                phases_t = np.array(phases[t - 24: t + 1])
                i = 24  # 24 due to size of temporary phase array (instead of directly modifying entire phase data)
                while (phases_t[i] != phases_t[i - 6]).any():
                    phases_t[i] = phases_t[i - 6]
                    i -= 1

                # Add columns to current instance
                for j in range(phases_t.shape[1]):
                    x_curr.append(phases_t[:, j])

            # Add to list of all instances
            x.append(x_curr)

            # Store output; include timestamps
            y.append([time[t + prediction_time], proton[t + prediction_time]])

        # Split train/test
        # Note that training targets do not need times, but test targets do for synchronization
        train, targets_train, test, targets_test =  self.train_test_split(x,y,self.config["data"]["train_split"])
        targets_test = {target[0]: float(target[1]) for target in targets_test}

        return np.array(train), np.array(targets_train)[:, 1].astype(np.float64), np.array(test), targets_test

    def evaluate(self,targets_test, predictions, event_times, data, path, train_time, display = False, save_plots = True):

        # First, get electron, high energy electron, and xray from data with timestamps for plots
        times = list(data["time"].values)
        electron = data["electron"].values
        electron_high = data["electron_high"].values

        # Convert to dictionaries
        electron_dict = {times[i]: electron[i] for i in range(len(electron) - len(targets_test), len(electron))}
        electron_high_dict = {times[i]: electron_high[i] for i in range(len(electron_high) - len(targets_test),
                                                                        len(electron_high))}

        # Verify that times synchronize with test targets
        if targets_test.keys() != electron_dict.keys() != electron_high_dict.keys():
            print("Timestamps are not synchronized, exiting program.")
            exit(0)

        # Begin evaluating events
        maes       = []
        o2p_lags   = []
        o2t_lags   = []
        ln10_lags  = []
        pes        = []
        r2s        = []
        f1s        = []   # (f1, tp, fp, fn, tn) per event, positive = flux above ln(10) (10 pfu)
        for i, event in enumerate(event_times):

            # Get event times
            onset = event[0]
            bg_before_onset = times[times.index(onset) - 36]
            threshold = event[1]
            peak = event[2]

            # Get relevant portions of each time series for plotting
            times_o2p = times[times.index(bg_before_onset): times.index(peak) + 1]
            targets_o2p = self.slice_dictionary_keys(targets_test, bg_before_onset, peak)
            predictions_o2p = self.slice_dictionary_keys(predictions, bg_before_onset, peak)
            electron_o2p = self.slice_dictionary_keys(electron_dict, bg_before_onset, peak)
            electron_high_o2p = self.slice_dictionary_keys(electron_high_dict, bg_before_onset, peak)

            # Event plot (skipped with save_plots=False, e.g. when only re-scoring saved predictions)
            if save_plots or display:
                PlotStuff.plot_event_forecast(times_o2p, targets_o2p, predictions_o2p, electron_o2p,
                                              electron_high_o2p, path, i + 1, display)

            # Now remove the 3 hours of background since they are not evaluated
            targets_o2p = self.slice_dictionary_keys(targets_test, onset, peak)
            predictions_o2p = self.slice_dictionary_keys(predictions, onset, peak)
            targets_o2t = self.slice_dictionary_keys(targets_test, onset, threshold)
            predictions_o2t = self.slice_dictionary_keys(predictions, onset, threshold)

            # Also convert from dictionary to numpy array; timestamps are no longer needed
            targets_o2p = np.array(list(targets_o2p.values()))
            predictions_o2p = np.array(list(predictions_o2p.values()))
            targets_o2t = np.array(list(targets_o2t.values()))
            predictions_o2t = np.array(list(predictions_o2t.values()))

            # Calculate stats
            maes.append(mae(targets_o2p, predictions_o2p, False))
            o2p_lags.append(x_axis_error(targets_o2p, predictions_o2p, False))
            o2t_lags.append(x_axis_error(targets_o2t, predictions_o2t, False))
            ln10_lags.append(lag_ln10(targets_o2p, predictions_o2p))
            pes.append(pe(targets_o2p, predictions_o2p))
            r2s.append(r2(targets_o2p, predictions_o2p))
            f1s.append(f1_counts(targets_o2p, predictions_o2p))

        # Output metrics per event
        if path:
            outfile = open(f'{path}/results.txt', 'w')
            for i in range(len(event_times)):
                outfile.write(f"Event {i + 1}\nMAE = {maes[i]: 0.3f}\nO2P lag = {o2p_lags[i]: 0.3f}\n"
                            f"O2T lag = {o2t_lags[i]: 0.3f}\nln10 lag = {ln10_lags[i]: 0.3f}\nPE = {pes[i]:0.3f}\nr2 = {r2s[i]: 0.3f}\n"
                            f"F1 = {f1s[i][0]: 0.3f}\nTP = {f1s[i][1]}\nFP = {f1s[i][2]}\nFN = {f1s[i][3]}\nTN = {f1s[i][4]}\n\n")
            outfile.write(f"Average MAE = {np.average(maes): 0.3f}\n")
            outfile.write(f"Average O2P lag = {np.average(o2p_lags): 0.3f}\n")
            outfile.write(f"Average O2T lag = {np.average(o2t_lags): 0.3f}\n")
            outfile.write(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}\n")
            outfile.write(f"Average PE = {np.average(pes): 0.3f}\n")
            outfile.write(f"Average r2 = {np.average(r2s): 0.3f}\n")
            outfile.write(f"Average F1 = {np.nanmean([f[0] for f in f1s]): 0.3f}\n")
            outfile.write(f"Train Time = {train_time}\n")
            outfile.close()

        
        # Output average metrics to standard output
        print(f"Average MAE = {np.average(maes): 0.3f}")
        print(f"Average O2P lag = {np.average(o2p_lags): 0.3f}")
        print(f"Average O2T lag = {np.average(o2t_lags): 0.3f}")
        print(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}")
        print(f"Average PE = {np.average(pes): 0.3f}")
        print(f"Average r2 = {np.average(r2s): 0.3f}")
        print(f"Average F1 = {np.nanmean([f[0] for f in f1s]): 0.3f}")
        print(f"Train Time = {train_time}\n")



    def run(self,data, prediction_time):
        return self.pair_input_output(data,self.config["data"]["use_phases"],prediction_time)        