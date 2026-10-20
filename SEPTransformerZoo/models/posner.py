
import numpy as np
from models.baseline_model import BaselineModel
import matplotlib.pyplot as plt
import matplotlib
from lib.torres.stats import *
import pandas as pd
from datetime import datetime


class Posner(BaselineModel):
    def __init__(self, config:dict):
        super().__init__(config)
        self.name = "posner"
        self.type = "baseline"
    
    def pair_input_output(self, data, prediction_time)-> tuple:
        """
        Pairs input windows of t=-24 to t=0 to model classes at t=prediction_time.
        :param electron: A list of electron intensities
        :param proton: A list of proton intensities
        :param prediction_time: The number of timesteps to predict ahead
        :return: The training and test sets, with their respective targets
        """

        # Set up data as <x, y> pairs
        x = []
        y = []
        time = list(data["time"].values)
        proton = data["proton"].values
        electron = data["electron"].values

        for t in range(24, len(proton) - prediction_time):
            max_rise = (electron[t - 12] - electron[t - 11]) / 5
            for interval in range(11, -1, -1):
                max_rise = max([max_rise, (electron[t - interval] - electron[t - interval + 1]) / 5])

            # Add to list of all inputs
            if max_rise <= 1e-2:
                max_rise = 1e-2
            elif max_rise >= 0.2:
                max_rise = 0.2
            if electron[t] > 8:
                electron_t = 8
            elif electron[t] < -3:
                electron_t = -3
            else:
                electron_t = electron[t]
            x.append([electron_t, np.log(max_rise)])  # log-log of slope

            # Add outputs based on future proton flux; include timestamps
            y.append([time[t + prediction_time], proton[t + prediction_time]])

        # Split train/test
        # Note that training targets do not need times, but test targets do for synchronization
        train, targets_train, test, targets_test = self.train_test_split(x, y, self.config["data"]["train_split"])
        targets_test = {target[0]: float(target[1]) for target in targets_test}

        return np.array(train, dtype=np.float64), np.array(targets_train)[:, 1].astype(np.float64), \
            np.array(test, dtype=np.float64), targets_test

    def train_model(self,train, targets_train, prediction_time, path):
        """
        Train the model.
        :param train: A 2D array of the training instances
        :param targets_train: A 1D array of the targets for the training instances
        :param prediction_time: Integer, number of timesteps to predict ahead (6 or 12)
        :param path: The directory in which to save the model plot
        :return: The trained model, as a 2D array, along with matrix containing parameter value ranges for each cell
        """

        # Get intensity ranges of each cell
        min_intensity = np.min(train[:, 0])
        max_intensity = np.max(train[:, 0])
        intensity_ranges = np.linspace(min_intensity, max_intensity, 19)  # +1 since every cell is a pair of values
        min_slope = np.min(train[:, 1])
        max_slope = np.max(train[:, 1])
        slope_ranges = np.linspace(min_slope, max_slope, 14)

        # Create 13 (slope) x 18 (intensity) matrix, with each cell containing two tuples:
        # (lower intensity, upper intensity), (lower slope, upper slope)
        range_matrix = [[0 for _ in range(len(slope_ranges) - 1)] for _ in range(len(intensity_ranges) - 1)]
        for i in range(len(intensity_ranges) - 1):
            for j in range(len(slope_ranges) - 1):
                range_matrix[i][j] = [(intensity_ranges[i], intensity_ranges[i + 1]),
                                    (slope_ranges[j], slope_ranges[j + 1])]
        range_matrix = np.flipud(range_matrix)  # Flip upside-down to match format of Posner paper

        # Plot number of instances per cell
        plt.hist2d(train[:, 1], train[:, 0], bins=[slope_ranges, intensity_ranges], norm=matplotlib.colors.LogNorm())
        plt.xlabel("Slope")
        plt.ylabel("Intensity")
        plt.colorbar()
        # plt.show()
        plt.close()

        # Train by averaging proton intensities of instances belonging to each cell
        model = [[0 for _ in range(range_matrix.shape[1])] for _ in range(range_matrix.shape[0])]
        num_instances_per_cell = [[0 for _ in range(range_matrix.shape[1])] for _ in range(range_matrix.shape[0])]
        for train_i, targets_train_i in zip(train, targets_train):
            matched = False

            # First, check for matches with max intensity, which will be in top row
            if train_i[0] == max_intensity:

                # Then find correct column
                for j in range(range_matrix.shape[1]):
                    if range_matrix[0][j][1][0] <= train_i[1] < range_matrix[0][j][1][1]:
                        model[0][j] += targets_train_i
                        num_instances_per_cell[0][j] += 1
                        matched = True
                        break

                # Special case for matching both maxes
                if train_i[1] == max_slope:
                    model[0][-1] += targets_train_i
                    num_instances_per_cell[0][-1] += 1
                    continue

            # Cells that don't match max intensity
            for i in range(range_matrix.shape[0]):
                if matched:
                    break
                for j in range(range_matrix.shape[1]):

                    # Cells that don't match max intensity or slope
                    if range_matrix[i][j][0][0] <= train_i[0] < range_matrix[i][j][0][1] and \
                            range_matrix[i][j][1][0] <= train_i[1] < range_matrix[i][j][1][1]:
                        model[i][j] += targets_train_i
                        num_instances_per_cell[i][j] += 1
                        matched = True
                        break

                # Cells that match max slope
                if train_i[1] == max_slope and range_matrix[i][-1][0][0] <= train_i[0] < range_matrix[i][-1][0][1]:
                    model[i][-1] += targets_train_i
                    num_instances_per_cell[i][-1] += 1
                    break

        # Average proton intensity per cell and display and return model
        model = np.array(model)
        num_instances_per_cell = np.array(num_instances_per_cell)
        # Cells with no training instances stay NaN (same result as plain division, without the 0/0 warning)
        model = np.divide(model, num_instances_per_cell, out=np.full_like(model, np.nan, dtype=np.float64),
                          where=num_instances_per_cell > 0)
        
        # Plot model
        plt.imshow(model)
        plt.title("Forecasting Matrix")
        plt.xlabel("Electron rise")
        xtick_labels = [f"{(slope_ranges[i] + slope_ranges[i + 1]) / 2:.02f}" for i in range(13)]
        plt.xticks(np.arange(13), xtick_labels)
        plt.ylabel("Electron intensity")
        ytick_labels = [f"{(intensity_ranges[i] + intensity_ranges[i + 1]) / 2:.02f}" for i in range(17, -1, -1)]
        plt.yticks(np.arange(18), ytick_labels)
        plt.colorbar()
        if path:
            plt.savefig(f"{path}/posner_model_t+{prediction_time}.png")
        plt.close()

        return model, range_matrix


    def predict(self,test, model, range_matrix):
        """
        Make predictions on test set using table obtained in training.
        :param test: A 2D array of the test instances
        :param model: A 2D array containing the proton intensities at each parameter value range
        :param range_matrix: A 2D array containing the value ranges of each parameter, each as a 2x2 matrix
        :return: A list of predicted proton intensities
        """

        # Get min/max parameter values in table
        # Indices are row, column, intensity (0) / slope (1), min (0) / max (1)
        min_intensity = range_matrix[-1][0][0][0]
        max_intensity = range_matrix[0][0][0][1]
        min_slope = range_matrix[0][0][1][0]
        max_slope = range_matrix[0][-1][1][1]

        predictions = []
        for test_i in test:

            # Clip to boundaries
            if test_i[0] < min_intensity:
                test_i[0] = min_intensity
            elif test_i[0] > max_intensity:
                test_i[0] = max_intensity
            if test_i[1] < min_slope:
                test_i[1] = min_slope
            elif test_i[1] > max_slope:
                test_i[1] = max_slope

            # Find where test instance parameters fall in model, and append predicted proton to predictions
            matched = False
            # First, check for matches with max intensity, which will be in top row
            if test_i[0] == max_intensity:

                # Then find correct column
                for j in range(range_matrix.shape[1]):
                    if range_matrix[0][j][1][0] <= test_i[1] < range_matrix[0][j][1][1]:
                        predictions.append(model[0][j])
                        matched = True
                        break

                # Special case for matching both maxes
                if test_i[1] == max_slope:
                    predictions.append(model[0][-1])
                    continue

            # Cells that don't match max intensity
            for i in range(range_matrix.shape[0]):
                if matched:
                    break
                for j in range(range_matrix.shape[1]):

                    # Cells that don't match max intensity or slope
                    if range_matrix[i][j][0][0] <= test_i[0] < range_matrix[i][j][0][1] and \
                            range_matrix[i][j][1][0] <= test_i[1] < range_matrix[i][j][1][1]:
                        predictions.append(model[i][j])
                        matched = True
                        break

                # Cells that match max slope
                if test_i[1] == max_slope and range_matrix[i][-1][0][0] <= test_i[0] < range_matrix[i][-1][0][1]:
                    predictions.append(model[i][-1])
                    break
            
        return np.array(predictions)


    # def evaluate(self,targets_test, predictions, event_times, data, path, display = False):

    #     # First, get electron, high energy electron, and xray from data with timestamps for plots
    #     times = list(data["time"].values)
    #     electron = data["electron"].values
    #     electron_high = data["electron_high"].values

    #     # Convert to dictionaries
    #     electron_dict = {times[i]: electron[i] for i in range(len(electron) - len(targets_test), len(electron))}
    #     electron_high_dict = {times[i]: electron_high[i] for i in range(len(electron_high) - len(targets_test),
    #                                                                     len(electron_high))}

    #     # Verify that times synchronize with test targets
    #     if targets_test.keys() != electron_dict.keys() != electron_high_dict.keys():
    #         print("Timestamps are not synchronized, exiting program.")
    #         exit(0)

    #     # Begin evaluating events
    #     maes = []
    #     o2p_lags = []
    #     o2t_lags = []
    #     ln10_lags = []
    #     for i, event in enumerate(event_times):

    #         # Get event times
    #         onset = event[0]
    #         bg_before_onset = times[times.index(onset) - 36]
    #         threshold = event[1]
    #         peak = event[2]

    #         # Get relevant portions of each time series for plotting
    #         times_o2p = times[times.index(bg_before_onset): times.index(peak) + 1]
    #         targets_o2p = self.slice_dictionary_keys(targets_test, bg_before_onset, peak)
    #         predictions_o2p = self.slice_dictionary_keys(predictions, bg_before_onset, peak)
    #         electron_o2p = self.slice_dictionary_keys(electron_dict, bg_before_onset, peak)
    #         electron_high_o2p = self.slice_dictionary_keys(electron_high_dict, bg_before_onset, peak)

    #         # Plot event
    #         fig, ax = plt.subplots()
    #         ax.set_ylabel('ln(Flux (/cc/s/sr))')
    #         ax.plot(list(targets_o2p.values()), '-b', label='Actual proton')
    #         ax.plot(list(predictions_o2p.values()), '-r', label='Predicted proton')
    #         ax.plot(list(electron_o2p.values()), '-m', label='Electron')
    #         ax.plot(list(electron_high_o2p.values()), '-y', label='High-energy electron')
    #         ax.plot([np.log(10)] * len(targets_o2p.values()), '--k')

    #         # Set x-axis to timestamps
    #         diff = int(ax.get_xticks()[1] - ax.get_xticks()[0])
    #         time_labels = [time[time.index('T') + 1: time.index('.')] for time in times_o2p[::diff]]
    #         if len(time_labels) == len(ax.get_xticks()) - 2:
    #             ax.set_xticks(ax.get_xticks()[1:-1])
    #         else:
    #             ax.set_xticks(ax.get_xticks()[1:-2])
    #         ax.set_xticklabels(time_labels, rotation='vertical')

    #         # Caption with date
    #         date_start = times_o2p[0][:times_o2p[0].index('T')]
    #         date_end = times_o2p[-1][:times_o2p[-1].index('T')]
    #         if date_start != date_end:
    #             ax.set_xlabel(f'Event from {date_start} to {date_end}')
    #         else:
    #             ax.set_xlabel(f'Event on {date_start}')

    #         fig.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
    #         fig.tight_layout()
    #         if path:
    #             plt.savefig(f'{path}/event{i + 1}.png')
    #         if display:
    #             plt.show()
    #         plt.close()

    #         # Now remove the 3 hours of background since they are not evaluated
    #         targets_o2p = self.slice_dictionary_keys(targets_test, onset, peak)
    #         predictions_o2p = self.slice_dictionary_keys(predictions, onset, peak)
    #         targets_o2t = self.slice_dictionary_keys(targets_test, onset, threshold)
    #         predictions_o2t = self.slice_dictionary_keys(predictions, onset, threshold)

    #         # Also convert from dictionary to numpy array; timestamps are no longer needed
    #         targets_o2p = np.array(list(targets_o2p.values()))
    #         predictions_o2p = np.array(list(predictions_o2p.values()))
    #         targets_o2t = np.array(list(targets_o2t.values()))
    #         predictions_o2t = np.array(list(predictions_o2t.values()))

    #         # Calculate stats
    #         maes.append(mae(targets_o2p, predictions_o2p, False))
    #         o2p_lags.append(x_axis_error(targets_o2p, predictions_o2p, False))
    #         o2t_lags.append(x_axis_error(targets_o2t, predictions_o2t, False))
    #         ln10_lags.append(lag_ln10(targets_o2p, predictions_o2p))

    #     # Output metrics per event
    #     if path:
    #         outfile = open(f'{path}/results.txt', 'w')
    #         for i in range(len(event_times)):
    #             outfile.write(f"Event {i + 1}\nMAE = {maes[i]: 0.3f}\nO2P lag = {o2p_lags[i]: 0.3f}\n"
    #                         f"O2T lag = {o2t_lags[i]: 0.3f}\nln10 lag = {ln10_lags[i]: 0.3f}\n\n")
    #         outfile.write(f"Average MAE = {np.average(maes): 0.3f}\n")
    #         outfile.write(f"Average O2P lag = {np.average(o2p_lags): 0.3f}\n")
    #         outfile.write(f"Average O2T lag = {np.average(o2t_lags): 0.3f}\n")
    #         outfile.write(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}\n")
    #         outfile.close()

    #     # Output average metrics to standard output
    #     print(f"Average MAE = {np.average(maes): 0.3f}")
    #     print(f"Average O2P lag = {np.average(o2p_lags): 0.3f}")
    #     print(f"Average O2T lag = {np.average(o2t_lags): 0.3f}")
    #     print(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}")


    def run(self,data:pd.DataFrame, prediction_time:float, event_times:pd.DataFrame,path:str) -> None:
        
        train, target_train, test, targets_test = self.pair_input_output(data, prediction_time)
        
        start_time = datetime.now()
        model, range_matrix = self.train_model(train,target_train,prediction_time, path)
        y_pred = self.predict(test,model, range_matrix)
        train_time = datetime.now() - start_time

        event_times_test = [event for event in event_times if event[0] in targets_test]
        target_times = list(targets_test.keys())
        predictions = {target_times[i]: y_pred[i] for i in range(len(y_pred))}

        np.savetxt(f"{path}/predictions.txt", y_pred, fmt='%.7f')

        self.evaluate(targets_test, predictions, event_times_test, data, path, train_time)
        return y_pred
